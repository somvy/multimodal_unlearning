import copy
import gc
import random

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

import wandb
from mm.trainer_utils import (
    forward_with_cache,
    get_batch_loss,
    has_lora_adapter,
    logits2probs,
    loss_needs_teacher,
    remove_image_tokens,
)


class MMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        # print("trainer: 19 inputs: ", inputs.keys())
        # print("trainer: 19 pixel_values: ", inputs["pixel_values"].shape)
        # print("trainer: 19 input_ids: ", inputs["input_ids"].shape)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    @torch.no_grad()
    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        outputs = model(**inputs)
        return (outputs.loss, outputs.logits, inputs["labels"])

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, *args, **kwargs):
        super()._maybe_log_save_evaluate(tr_loss, grad_norm, model, *args, **kwargs)

        # Log generation examples
        if self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0:
            self._log_generation_examples(model)

        # gc.collect()
        # torch.cuda.empty_cache()
        # gc.collect()

    def _log_generation_examples(self, model: torch.nn.Module, num_examples: int = 3):
        return  # Disable generation examples for now

        def prepare_input(item, has_image: bool):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": item["question"]},
                    ],
                },
            ]
            if has_image:
                conversation[0]["content"].insert(0, {"type": "image"})
            formatted_question = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            return formatted_question

        # Get a small batch of evaluation data
        eval_dataset = self.get_eval_dataloader().dataset
        num_samples = len(eval_dataset)
        random_indices = random.sample(range(num_samples), num_examples)
        batch = [eval_dataset[idx] for idx in random_indices]

        texts = [prepare_input(item, item.get("image", None) is not None) for item in batch]
        images = [item["image"] for item in batch if item.get("image", None) is not None]
        inputs = self.processor(
            text=texts,
            images=images if images else None,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        ).to(model.device)

        # # Move batch to the same device as the model
        # inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,  # Adjust as needed
                num_return_sequences=1,
                do_sample=True,
            )

        # Decode the generated outputs
        generated_texts = self.processor.batch_decode(outputs, skip_special_tokens=True)
        input_texts = self.processor.batch_decode(inputs["input_ids"], skip_special_tokens=True)
        answers = [item["answer"] for item in batch]

        logs = {}
        data = []
        for i, (input_text, generated_text, answer) in enumerate(zip(input_texts, generated_texts, answers)):
            logs[f"generation_example_{i}/input"] = input_text
            logs[f"generation_example_{i}/output"] = generated_text
            data.append([input_text, generated_text, answer])
        columns = ["Input", "Generated Output", "Answer"]
        generation_table = wandb.Table(data=data, columns=columns)
        # Log the table to wandb
        if wandb.run is not None:
            wandb.log({"Generation Examples": generation_table}, step=self.state.global_step)
        self.log(logs)


class MMTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop("forget_loss").lower()
        self.teacher_model = kwargs.pop("teacher_model")
        self.loss_beta = kwargs.pop("loss_beta")
        self.l1_lambda = kwargs.pop("l1_lambda")
        self.l0_lambda = kwargs.pop("l0_lambda")
        self.l_norm_from = kwargs.pop("l_norm_from")
        self.loss_args = kwargs.pop("loss_args", {})

        if self.l1_lambda and self.l1_lambda != 0 and self.l_norm_from not in ("zero", "init"):
            raise ValueError(f"Invalid l_norm_from {self.l_norm_from}, should be 'zero' or 'init'")

        super(MMTrainerForgetting, self).__init__(*args, **kwargs)

        if (loss_needs_teacher(self.loss_type) or self.l1_lambda != 0 or self.l0_lambda != 0) and self.is_deepspeed_enabled:
            self.teacher_model = self.e_prepare_deepspeed(self.teacher_model)

    def e_prepare_deepspeed(self, model):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = copy.deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes) if getattr(model.config, "hidden_sizes", None) else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        config_kwargs["optimizer"] = {"type": None}
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        # set the gradients to false for every parameter
        for param in model.parameters():
            param.requires_grad = False

        return model

    @staticmethod
    def to_device(device, inputs):
        return {k: v.to(device) for k, v in inputs.items()}

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        model.device = next(model.parameters()).device
        retain_inputs = self.to_device(model.device, inputs["retain"])

        if "idk" in inputs:
            idk_inputs = self.to_device(model.device, inputs["idk"])
        if "forget" in inputs:
            forget_inputs = self.to_device(model.device, inputs["forget"])
        if "random" in inputs:
            random_inputs = self.to_device(model.device, inputs["random"])

        if self.loss_type == "retain_ft":
            retain_outputs = model(**retain_inputs)
            loss = retain_outputs.loss

        elif self.loss_type == "grad_ascent":
            forget_outputs = model(**forget_inputs)
            loss = -1 * forget_outputs.loss

        elif self.loss_type.startswith("grad_diff"):
            forget_outputs = model(**forget_inputs)
            retain_outputs = model(**retain_inputs)

            if "forget_ce" in self.loss_type:
                forget_loss = forget_outputs.loss

            elif "forget_entropy" in self.loss_type:
                forget_probs = logits2probs(forget_outputs.logits, log=False)
                forget_loss = torch.sum(forget_probs * torch.log(forget_probs))

            elif "forget_kl" in self.loss_type:
                with torch.no_grad():
                    oracle_forget_outputs = self.teacher_model(**forget_inputs)
                oracle_forget_probs = logits2probs(oracle_forget_outputs.logits)
                forget_probs = logits2probs(forget_outputs.logits)
                forget_loss = nn.functional.kl_div(
                    forget_probs,
                    oracle_forget_probs,
                    reduction="batchmean",
                    log_target=True,
                )
            else:
                raise ValueError(f"Invalid loss type on forget {self.loss_type}")

            if "retain_ce" in self.loss_type:
                retain_loss = retain_outputs.loss

            elif "retain_kl" in self.loss_type:
                with torch.no_grad():
                    oracle_retain_outputs = self.teacher_model(**retain_inputs)
                oracle_retain_probs = logits2probs(oracle_retain_outputs.logits)
                retain_probs = logits2probs(retain_outputs.logits)
                retain_loss = nn.functional.kl_div(
                    retain_probs,
                    oracle_retain_probs,
                    reduction="batchmean",
                    log_target=True,
                )
            else:
                raise ValueError(f"Invalid loss type on retain {self.loss_type}")

            loss = -1 * self.loss_beta * forget_loss + retain_loss

        elif self.loss_type == "scrub":
            forget_outputs = model(**forget_inputs)
            forget_probs = logits2probs(forget_outputs.logits)
            with torch.no_grad():
                oracle_forget_outputs = self.teacher_model(**forget_inputs)
            oracle_forget_probs = logits2probs(oracle_forget_outputs.logits)
            kl_forget_loss = nn.functional.kl_div(
                oracle_forget_probs,
                forget_probs,
                reduction="batchmean",
                log_target=True,
            )

            retain_outputs = model(**retain_inputs)
            retain_probs = logits2probs(retain_outputs.logits)
            with torch.no_grad():
                oracle_retain_outputs = self.teacher_model(**retain_inputs)
            oracle_retain_probs = logits2probs(oracle_retain_outputs.logits)

            kl_retain_loss = nn.functional.kl_div(
                oracle_retain_probs,
                retain_probs,
                reduction="batchmean",
                log_target=True,
            )

            loss = -1 * self.loss_beta * kl_forget_loss + kl_retain_loss + retain_outputs.loss

        elif self.loss_type == "kl":
            forget_outputs = model(**forget_inputs)
            forget_loss = -1 * forget_outputs.loss

            with torch.no_grad():
                oracle_retain_outputs = self.teacher_model(**retain_inputs)
            oracle_retain_probs = logits2probs(oracle_retain_outputs.logits)

            retain_outputs = model(**retain_inputs)
            retain_probs = logits2probs(retain_outputs.logits)

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(
                retain_probs,
                oracle_retain_probs,
                reduction="batchmean",
                log_target=True,
            )
            loss = forget_loss + retain_loss

        elif self.loss_type == "llmu":
            forget_outputs = model(**forget_inputs)
            forget_loss = -1 * forget_outputs.loss

            idk_outputs = model(**idk_inputs)

            random_loss = idk_outputs.loss

            retain_outputs = model(**retain_inputs)
            with torch.no_grad():
                oracle_retain_outputs = self.teacher_model(**retain_inputs)

            oracle_retain_probs = logits2probs(oracle_retain_outputs.logits)
            retain_probs = logits2probs(retain_outputs.logits)

            retain_loss = nn.functional.kl_div(
                oracle_retain_probs,
                retain_probs,
                reduction="batchmean",
                log_target=True,
            )
            loss = forget_loss + retain_loss + random_loss

        elif self.loss_type == "rmu":
            # !python3 -m rmu.unlearn --model_name mistralai/Mixtral-8x7B-Instruct-v0.1  --param_ids 7 --steering_coeffs 300,300 --alpha 1600,1600  --output_dir models/mixtral_rmu
            if self.is_deepspeed_enabled:
                target_module = model.module.model.language_model.model.layers[7]
                oracle_target_module = self.teacher_model.module.language_model.model.layers[7]
            else:
                target_module = model.language_model.model.layers[7]
                oracle_target_module = self.teacher_model.language_model.model.layers[7]

            forget_activations = forward_with_cache(model, forget_inputs, target_module, no_grad=False)  # .to(model.device)

            if self.__dict__.get("control_vec", None) is None:
                hidden_size = model.config.hidden_size if not self.is_deepspeed_enabled else model.module.model.config.hidden_size
                batch_size = forget_activations.shape[0]
                rand_vec = torch.rand(batch_size, 1, hidden_size, dtype=forget_activations.dtype, device=forget_activations.device)
                self.control_vec = rand_vec / torch.norm(rand_vec) * 300

            forget_loss = torch.nn.functional.mse_loss(forget_activations.mean(dim=1), self.control_vec)
            forget_loss *= self.loss_beta

            retain_activations = forward_with_cache(model, retain_inputs, target_module, no_grad=False).to(model.device)
            oracle_retain_activations = forward_with_cache(self.teacher_model, retain_inputs, oracle_target_module, no_grad=True).to(model.device)
            retain_loss = torch.nn.functional.mse_loss(retain_activations, oracle_retain_activations)

            loss = forget_loss + retain_loss

        # elif self.loss_type == "dpo_orig":

        #     policy_chosen_logps =

        # pi_logratios = policy_chosen_logps - policy_rejected_logps
        # ref_logratios = reference_chosen_logps - reference_rejected_logps

        # logits = pi_logratios - ref_logratios

        # # if self.loss_type == "sigmoid":  # rkl
        # losses = -F.logsigmoid(self.beta * logits)
        # chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        # rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        # return losses, chosen_rewards, rejected_rewards
        elif self.loss_type == "npo":
            forget_outputs = model(**forget_inputs)
            with torch.no_grad():
                oracle_forget_outputs = self.teacher_model(**forget_inputs)

            oracle_forget_probs = logits2probs(oracle_forget_outputs.logits, log=False)
            forget_probs = logits2probs(forget_outputs.logits, log=False)

            pi_ratios = forget_probs / (oracle_forget_probs + 1e-8)

            loss = (2 / self.loss_beta) * torch.mean(torch.log(1 + torch.pow(pi_ratios, self.loss_beta)))

        elif self.loss_type == "idk":
            retain_outputs = model(**retain_inputs)
            idk_outputs = model(**idk_inputs)
            loss = retain_outputs.loss + idk_outputs.loss
        elif self.loss_type == "eco_ft":
            forget_outputs = model(**forget_inputs)
            retain_outputs = model(**retain_inputs)
            loss = self.loss_beta * forget_outputs.loss + retain_outputs.loss

        elif self.loss_type == "dpo":
            idk_outputs = model(**idk_inputs)
            forget_outputs = model(**forget_inputs)

            with torch.no_grad():
                idk_outputs_oracle = self.teacher_model(**idk_inputs)
                forget_outputs_oracle = self.teacher_model(**forget_inputs)
            image_token_index = model.config.image_token_index if not self.is_deepspeed_enabled else model.module.model.config.image_token_index

            idk_logits_oracle = remove_image_tokens(idk_inputs["input_ids"], idk_outputs_oracle.logits, image_token_index)

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_inputs["labels"])

            forget_logits_oracle = remove_image_tokens(
                forget_inputs["input_ids"],
                forget_outputs_oracle.logits,
                image_token_index,
            )
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_inputs["labels"])

            idk_logits = remove_image_tokens(idk_inputs["input_ids"], idk_outputs.logits, image_token_index)
            idk_loss_current = -1 * get_batch_loss(idk_logits, idk_inputs["labels"])

            forget_logits = remove_image_tokens(forget_inputs["input_ids"], forget_outputs.logits, image_token_index)
            forget_loss_current = -1 * get_batch_loss(forget_logits, forget_inputs["labels"])

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            #
            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            loss = -pi_logratios.mean()
            loss = -idk_loss_current.mean()

            outputs = forget_outputs

        # elif self.loss_type == "multi_delete":
        #     # raise NotImplementedError("multi_delete is not implemented yet")
        #     # get inputs, calculate cliploss between them
        #     current_model = model.module.model if self.is_deepspeed_enabled else model
        #     image_outputs = current_model.vision_tower(pixel_values, output_hidden_states=True)
        #     image_embs = current_model.multi_modal_projector(
        #         image_outputs.hidden_states[model.config.vision_feature_layer][:, 1:]
        #     )

        #     # from which token extract embs
        #     # text_embs =

        elif self.loss_type == "sku":
            # based on https://github.com/franciscoliu/SKU/blob/main/harmful_unlearn/unlearn_harm_new.py

            bad_loss = model(**forget_inputs).loss

            # the preprocessing is done in the dataset.py::MMMixedForgetDataset
            # random_inputs contains list of lists, each contains K inputs
            # total shape : batch_size x K samples x seq_len
            for k, v in random_inputs.items():
                random_inputs[k] = v.transpose(0, 1).to(model.device)
            # FIXME: here may be CUDA memory overflow, due to the larger number of samples
            K = random_inputs["input_ids"].shape[0]

            random_loss = 0.0

            for i in range(K):
                inputs = {k: v[i] for k, v in random_inputs.items()}
                random_loss += model(**inputs).loss

            with torch.no_grad():
                teacher_retain_logprobs = logits2probs(self.teacher_model(**retain_inputs).logits)
            retain_logprobs = logits2probs(model(**retain_inputs).logits)

            normal_loss = -1 * nn.functional.kl_div(
                retain_logprobs,
                teacher_retain_logprobs,
                reduction="batchmean",
                log_target=True,
            )

            loss = (
                self.loss_args["bad_weight"] * bad_loss
                + self.loss_args["random_weight"] * random_loss
                + self.loss_args["normal_weight"] * normal_loss
            )

        else:
            raise ValueError(f"Invalid loss type {self.loss_type}")

        if (self.l1_lambda is not None and self.l1_lambda != 0) or (self.l0_lambda is not None and self.l0_lambda != 0):
            params = []
            if self.l_norm_from == "init":
                if has_lora_adapter(model):
                    # include only the trainable lora parameters
                    for param in model.parameters():
                        if param.requires_grad:
                            params.append(param.view(-1))

                else:
                    # just calculate the difference with original model
                    if self.teacher_model is None:
                        raise ValueError("Teacher model is required for L1\L0 regularization during training without LORA!")
                    for param, teacher_param in zip(model.parameters(), self.teacher_model.parameters()):
                        assert param.shape == teacher_param.shape
                        if param.requires_grad:
                            params.append((param - teacher_param.to(model.device)).view(-1))

            elif self.l_norm_from == "zero":
                if has_lora_adapter(model):
                    raise ValueError("L1\L0 regularization is not supported for LORA!")

                for param in model.parameters():
                    if param.requires_grad:
                        params.append(param.view(-1))

            if self.l1_lambda is not None and self.l1_lambda != 0:
                loss = loss + self.l1_lambda * torch.norm(torch.cat(params), p=1)

            if self.l0_lambda is not None and self.l0_lambda != 0:
                loss = loss + self.l0_lambda * torch.norm(torch.cat(params), p=0)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
