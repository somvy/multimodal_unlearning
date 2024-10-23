import os
import re
from functools import partial

import torch
import hydra
import transformers
from omegaconf import OmegaConf
from transformers import AutoProcessor, AutoModelForPreTraining, AutoConfig, set_seed
from pathlib import Path
from dotenv import load_dotenv

from peft import LoraConfig, get_peft_model, PeftModel

from mm.dataset import MMMixedForgetDataset, mm_forget_data_collator_preprocessor
from mm.trainer import MMTrainerForgetting
from mm.trainer_utils import loss_needs_oracle
from utils import get_model_identifiers_from_yaml, find_all_linear_names, print_trainable_parameters, freeze_params


@hydra.main(version_base=None, config_path="../config/mm", config_name="forget")
def main(cfg):
    load_dotenv()
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    print(f"num_devices: {num_devices}")

    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

    else:
        local_rank = 0
        device_map = None

    set_seed(cfg.seed)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    print("######################")
    print("Saving to: ", cfg.save_dir)
    print("######################")

    if local_rank == 0:
        if os.path.exists(cfg.save_dir):
            print("Directory already exists")
            if not cfg.overwrite_dir:
                exit()

        Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

        with open(f"{cfg.save_dir}/config.yaml", "w") as file:
            OmegaConf.save(cfg, file)

    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    processor.do_pad = True
    max_length = 1024

    torch_format_dataset = MMMixedForgetDataset(
        forget_data_path=cfg.forget_data_path,
        retain_data_path=cfg.retain_data_path,
        retain_split=cfg.retain_split,
        forget_split=cfg.forget_split,
        forget_loss=cfg.forget_loss,
    )

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    steps_per_epoch = len(torch_format_dataset) // (batch_size * gradient_accumulation_steps * num_devices)
    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (batch_size * gradient_accumulation_steps * num_devices)
    print(f"max_steps: {max_steps}")

    # first get the base model architecture
    # if there is a pytorch*.bin file in the model path, then load that. use regex there can be anythign in between pytorch and .bin
    # check that folder contatins a pytorch model
    path_found = any(
        re.search("pytorch.*\.bin", file.name) or re.search("model-*\.safetensors", file.name) for file in Path(cfg.model_path).glob("*")
    )

    oracle_model = None

    if path_found:
        config = AutoConfig.from_pretrained(model_id)

        print("Loading from checkpoint")
        model = AutoModelForPreTraining.from_pretrained(
            cfg.model_path,
            config=config,
            use_flash_attention_2=model_cfg["flash_attention2"] == "true",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        if cfg.l1_lambda != 0 or cfg.l0_lambda != 0 or loss_needs_oracle(cfg.forget_loss):
            oracle_model = AutoModelForPreTraining.from_pretrained(
                cfg.model_path,
                config=config,
                use_flash_attention_2=model_cfg["flash_attention2"] == "true",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
            # needed for deepspeed
            oracle_model.config.hidden_size = oracle_model.config.text_config.hidden_size

    else:
        print("Loading after merge and unload")
        model = AutoModelForPreTraining.from_pretrained(
            model_id, use_flash_attention_2=model_cfg["flash_attention2"] == "true", torch_dtype=torch.bfloat16, device_map=device_map
        )
        # now use the checkpoint to add the LoRA modules
        model = PeftModel.from_pretrained(model, model_id=cfg.model_path)
        # save this as a standard model so that we can again do PEFT style finetuneing from scratch
        model = model.merge_and_unload()
        # save the model for next time
        model.save_pretrained(cfg.model_path)

    freeze_params(model.vision_tower)
    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    # needed for deepspeed
    model.config.hidden_size = model.config.text_config.hidden_size

    # now we have a HuggingFace model
    if model_cfg["gradient_checkpointing"] == "true":
        model.gradient_checkpointing_enable()

    if cfg.LoRA.r != 0 and (cfg.l_norm_from != "zero" and cfg.l1_lambda != 0):
        config = LoraConfig(
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=cfg.LoRA.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    training_args = transformers.TrainingArguments(
        disable_tqdm=False,
        remove_unused_columns=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        # gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_steps=max(1, steps_per_epoch),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        # max_grad_norm=cfg.max_grad_norm,
        # bf16_full_eval=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=f"{cfg.save_dir}/logs",
        output_dir=cfg.save_dir,
        optim="adamw_bnb_8bit",
        # optim="adamw_torch_4bit",
        save_strategy="steps" if cfg.save_ckpts else "no",
        save_steps=steps_per_epoch,
        save_only_model=True,
        ddp_find_unused_parameters=True,
        # deepspeed="config/ds_config.json",
        weight_decay=cfg.weight_decay,
        eval_steps=steps_per_epoch,
        # eval_strategy="steps" if cfg.eval_while_train else "no",
        seed=cfg.seed,
    )

    mm_data_collator = partial(mm_forget_data_collator_preprocessor, processor=processor, max_length=max_length)
    trainer = MMTrainerForgetting(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        compute_metrics=None,  # the callback for computing metrics, None in this case since you're doing it in your callback
        # callbacks=[GlobalStepDeletionCallback],
        args=training_args,
        data_collator=mm_data_collator,
        oracle_model=oracle_model,
        forget_loss=cfg.forget_loss,
        loss_beta=cfg.loss_beta,
        l1_lambda=cfg.l1_lambda,
        l0_lambda=cfg.l0_lambda,
        l_norm_from=cfg.l_norm_from,
    )
    # model.config.use_cache = (False ) # silence the warnings. Please re-enable for inference!

    trainer.train()

    # save the tokenizer
    if cfg.save_model:
        model.save_pretrained(cfg.save_dir)
        processor.save_pretrained(cfg.save_dir)

    # delete all "global_step*" files in the save_dir/checkpoint-*/ directories
    if local_rank == 0:
        import shutil

        for file in Path(cfg.save_dir).glob("checkpoint-*"):
            for global_step_dir in file.glob("global_step*"):
                # delete the directory
                shutil.rmtree(global_step_dir)


if __name__ == "__main__":
    main()
