import os
import sys

import torch
from transformers import (
    set_seed,
    AutoProcessor,
    AutoModelForPreTraining,
    TrainingArguments,
)
import hydra
from peft import LoraConfig, get_peft_model
from pathlib import Path
from omegaconf import OmegaConf
from functools import partial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mm.trainer import MMTrainer
from mm.dataset import mm_data_collator_preprocessor, MMMixedDataset
from utils import (
    get_model_identifiers_from_yaml,
    find_all_linear_names,
)

# os.environ["WANDB_DISABLED"] = "true"


@hydra.main(version_base=None, config_path="../config/mm", config_name="finetune")
def main(cfg):
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
    else:
        device_map = "auto"
    set_seed(cfg.seed)
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    # save the cfg file
    # if master process
    if os.environ.get("LOCAL_RANK") is None or local_rank == 0:
        with open(f"{cfg.save_dir}/cfg.yaml", "w") as f:
            OmegaConf.save(cfg, f)

    processor = AutoProcessor.from_pretrained(model_id)
    # processor.image_processor.do_resize = True
    # processor.image_processor.size = {"shortest_edge": 224}
    # processor.image_processor.do_center_crop = True

    processor.tokenizer.padding_side = "left"
    processor.do_pad = True
    max_length = 1024
    torch_format_dataset = MMMixedDataset(
        data_path=cfg.data_path,
        split=cfg.split,
    )

    batch_size = cfg.batch_size
    gradient_accumulation_steps = cfg.gradient_accumulation_steps
    # --nproc_per_node gives the number of GPUs per = num_devices. take it from torchrun/os.environ
    num_devices = int(os.environ.get("WORLD_SIZE", 1))
    print(f"num_devices: {num_devices}")

    max_steps = int(cfg.num_epochs * len(torch_format_dataset)) // (
        batch_size * gradient_accumulation_steps * num_devices
    )
    # pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    # max_steps=5
    print(f"max_steps: {max_steps}")
    training_args = TrainingArguments(
        remove_unused_columns=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=model_cfg["gradient_checkpointing"] == "true",
        # warmup_steps=max(1, max_steps//10),
        warmup_steps=max(1, max_steps // cfg.num_epochs),
        max_steps=max_steps,
        learning_rate=cfg.lr,
        bf16=True,
        # bf16_full_eval=True,
        logging_steps=max(1, max_steps // 20),
        logging_dir=f"{cfg.save_dir}/logs",
        output_dir=cfg.save_dir,
        optim="adamw_bnb_8bit",
        save_steps=max_steps,
        save_only_model=True,
        eval_strategy="no",
        deepspeed="config/ds_config.json",
        weight_decay=cfg.weight_decay,
        ddp_find_unused_parameters=True,
        seed=cfg.seed,
        report_to=["wandb"],
    )

    model = AutoModelForPreTraining.from_pretrained(
        model_id,
        use_flash_attention_2=model_cfg["flash_attention2"] == "true",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    )
    # freeze_params(model.vision_tower)
    # needed for deepspeed
    model.config.hidden_size = model.config.text_config.hidden_size

    # Hot fix for https://discuss.huggingface.co/t/help-with-llama-2-finetuning-setup/50035
    model.generation_config.do_sample = True
    if cfg.LoRA.r != 0:
        config = LoraConfig(
            r=cfg.LoRA.r,
            lora_alpha=cfg.LoRA.alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=cfg.LoRA.dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.enable_input_require_grads()

    mm_data_collator = partial(
        mm_data_collator_preprocessor,
        processor=processor,
        max_length=max_length,
    )
    trainer = MMTrainer(
        model=model,
        tokenizer=processor.tokenizer,
        train_dataset=torch_format_dataset,
        eval_dataset=torch_format_dataset,
        args=training_args,
        data_collator=mm_data_collator,
    )
    # need to set processor manually because it is not used in trainer init
    # it is used in _log_generation_examples
    trainer.processor = processor

    # silence the warnings. Please re-enable for inference!
    model.config.use_cache = False
    #
    trainer.train(resume_from_checkpoint=(cfg.resume_from_checkpoint == "true"))


    # save the model
    if cfg.LoRA.r != 0:
        model = model.merge_and_unload()

    model.save_pretrained(cfg.save_dir)
    processor.save_pretrained(cfg.save_dir)


if __name__ == "__main__":
    main()
