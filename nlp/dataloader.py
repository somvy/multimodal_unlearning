import copy

import deepspeed
import torch
import torch.nn.functional as F
from data_module import get_batch_loss
from torch import nn
from transformers import Trainer


def printll(name, inp):
    # print list with 4 decimal for each item
    print(name, [round(x, 4) for x in inp])


def has_lora_adapter(model):
    return any(True for name, param in model.named_parameters() if "lora" in name)


def loss_needs_oracle(loss_type: str):
    return "KL" in loss_type or loss_type in ("dpo", "scrub", "RMU", "LLMU")


def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)
    inputs = {k: v.to(model.device) for k, v in zip(["input_ids", "labels", "attention_mask"], inputs)}

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()

    return cache[0]


def logits2probs(logits, log=True):
    if log:
        probs = F.log_softmax(logits, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)
    return probs.view(-1, logits.shape[-1])


class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction="batchmean", log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        p, q = (
            p.view(-1, p.size(-1)).log_softmax(-1),
            q.view(-1, q.size(-1)).log_softmax(-1),
        )
        m = 0.5 * (p + q)
        return 0.5 * (self.kl(m, p) + self.kl(m, q))


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids, labels, attention_mask = inputs
        # forward pass
        outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
        # logits = outputs.get("logits")
        loss = outputs.loss
        # # compute custom loss (suppose one has 3 labels with different weights)
        # loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0], device=model.device))
        # loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = inputs
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)


class CustomTrainerForgetting(Trainer):
    def __init__(self, *args, **kwargs):
        self.loss_type = kwargs.pop("forget_loss")
        self.oracle_model = kwargs.pop("oracle_model")
        self.loss_beta = kwargs.pop("loss_beta")
        self.l1_lambda = kwargs.pop("l1_lambda")
        self.l0_lambda = kwargs.pop("l0_lambda")

        super(CustomTrainerForgetting, self).__init__(*args, **kwargs)
        if (loss_needs_oracle(self.loss_type) or self.l1_lambda != 0 or self.l0_lambda != 0) and self.is_deepspeed_enabled:
            self.oracle_model = self.e_prepare_deepspeed(self.oracle_model)

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
        return (inputs[0].to(device), inputs[1].to(device), inputs[2].to(device))

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs, retain_inputs, *idk_inputs = inputs
        forget_input_ids, forget_labels, forget_attention_mask = self.to_device(model.device, forget_inputs)
        retain_input_ids, retain_labels, retain_attention_mask = self.to_device(model.device, retain_inputs)
        if idk_inputs:
            idk_input_ids, idk_labels, idk_attention_mask = self.to_device(model.device, idk_inputs[0])

        if self.loss_type == "retain_ft":
            retain_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )
            loss = retain_outputs.loss

        elif self.loss_type == "grad_ascent":
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )
            loss = -1 * forget_outputs.loss

        elif self.loss_type.startswith("grad_diff"):
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )
            retain_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )

            if "forget_ce" in self.loss_type:
                forget_loss = forget_outputs.loss

            elif "forget_entropy" in self.loss_type:
                forget_probs = logits2probs(forget_outputs.logits, log=False)
                forget_loss = torch.sum(forget_probs * torch.log(forget_probs))

            elif "forget_KL" in self.loss_type:
                with torch.no_grad():
                    oracle_forget_outputs = self.oracle_model(
                        forget_input_ids,
                        labels=forget_labels,
                        attention_mask=forget_attention_mask,
                    )
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

            elif "retain_KL" in self.loss_type:
                with torch.no_grad():
                    oracle_retain_outputs = self.oracle_model(
                        retain_input_ids,
                        labels=retain_labels,
                        attention_mask=retain_attention_mask,
                    )
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
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )
            forget_probs = logits2probs(forget_outputs.logits)
            with torch.no_grad():
                oracle_forget_outputs = self.oracle_model(
                    forget_input_ids,
                    labels=forget_labels,
                    attention_mask=forget_attention_mask,
                )
            oracle_forget_probs = logits2probs(oracle_forget_outputs.logits)
            kl_forget_loss = nn.functional.kl_div(
                oracle_forget_probs,
                forget_probs,
                reduction="batchmean",
                log_target=True,
            )

            retain_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )
            retain_probs = logits2probs(retain_outputs.logits)
            with torch.no_grad():
                oracle_retain_outputs = self.oracle_model(
                    retain_input_ids,
                    labels=retain_labels,
                    attention_mask=retain_attention_mask,
                )
            oracle_retain_probs = logits2probs(oracle_retain_outputs.logits)

            kl_retain_loss = nn.functional.kl_div(
                oracle_retain_probs,
                retain_probs,
                reduction="batchmean",
                log_target=True,
            )

            loss = -1 * self.loss_beta * kl_forget_loss + kl_retain_loss + retain_outputs.loss

        elif self.loss_type == "KL":
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )
            forget_loss = -1 * forget_outputs.loss

            with torch.no_grad():
                oracle_retain_outputs = self.oracle_model(
                    retain_input_ids,
                    labels=retain_labels,
                    attention_mask=retain_attention_mask,
                )
            oracle_retain_probs = logits2probs(oracle_retain_outputs.logits)

            retain_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )
            retain_probs = logits2probs(retain_outputs.logits)

            # minimum KL divergence
            retain_loss = nn.functional.kl_div(
                retain_probs,
                oracle_retain_probs,
                reduction="batchmean",
                log_target=True,
            )
            loss = forget_loss + retain_loss

        elif self.loss_type == "LLMU":
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )
            forget_loss = -1 * forget_outputs.loss

            idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)

            random_loss = idk_outputs.loss

            retain_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )

            with torch.no_grad():
                oracle_retain_outputs = self.oracle_model(
                    retain_input_ids,
                    labels=retain_labels,
                    attention_mask=retain_attention_mask,
                )

            oracle_retain_probs = logits2probs(oracle_retain_outputs.logits)
            retain_probs = logits2probs(retain_outputs.logits)

            retain_loss = nn.functional.kl_div(
                oracle_retain_probs,
                retain_probs,
                reduction="batchmean",
                log_target=True,
            )
            loss = forget_loss + retain_loss + random_loss

        elif self.loss_type == "RMU":
            # !python3 -m rmu.unlearn --model_name mistralai/Mixtral-8x7B-Instruct-v0.1  --param_ids 7 --steering_coeffs 300,300 --alpha 1600,1600  --output_dir models/mixtral_rmu
            if self.is_deepspeed_enabled:
                updated_module = model.module.model.model.layers[7]
            else:
                updated_module = model.model.layers[7]
            forget_activations = forward_with_cache(model, forget_inputs, updated_module, no_grad=False).to(model.device)
            hidden_size = model.config.hidden_size if not self.is_deepspeed_enabled else model.module.model.config.hidden_size
            rand_vec = torch.rand(
                1,
                1,
                hidden_size,
                dtype=forget_activations.dtype,
                device=forget_activations.device,
            )
            control_vec = rand_vec / torch.norm(rand_vec) * 300
            forget_loss = torch.nn.functional.mse_loss(forget_activations, control_vec)
            forget_loss *= self.loss_beta

            if self.is_deepspeed_enabled:
                oracle_updated_module = self.oracle_model.module.model.layers[7]
            else:
                oracle_updated_module = self.oracle_model.model.layers[7]
            retain_activations = forward_with_cache(model, retain_inputs, updated_module, no_grad=False).to(model.device)

            oracle_retain_activations = forward_with_cache(self.oracle_model, retain_inputs, oracle_updated_module, no_grad=True).to(model.device)
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
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )
            with torch.no_grad():
                oracle_forget_outputs = self.oracle_model(
                    forget_input_ids,
                    labels=forget_labels,
                    attention_mask=forget_attention_mask,
                )
            oracle_forget_probs = logits2probs(oracle_forget_outputs.logits, log=False)
            forget_probs = logits2probs(forget_outputs.logits, log=False)

            pi_ratios = torch.log(forget_probs / oracle_forget_probs)

            loss = 2 / self.loss_beta * torch.mean(torch.log(1 + pi_ratios**self.loss_beta))

        elif self.loss_type == "idk":
            # "i dont know" inputs here are in forget inputs
            # then concatenate the inputs. single forward pass is much more efficient
            input_ids = torch.cat((forget_input_ids, retain_input_ids), dim=0)
            labels = torch.cat((forget_labels, retain_labels), dim=0)
            attention_mask = torch.cat((forget_attention_mask, retain_attention_mask), dim=0)
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs.loss
        elif self.loss_type == "eco_ft":
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )

            retain_outputs = model(
                retain_input_ids,
                labels=retain_labels,
                attention_mask=retain_attention_mask,
            )
            loss = self.loss_beta * forget_outputs.loss + retain_outputs.loss

        elif self.loss_type == "dpo":
            idk_outputs = model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
            forget_outputs = model(
                forget_input_ids,
                labels=forget_labels,
                attention_mask=forget_attention_mask,
            )

            with torch.no_grad():
                idk_outputs_oracle = self.oracle_model(idk_input_ids, labels=idk_labels, attention_mask=idk_attention_mask)
                forget_outputs_oracle = self.oracle_model(
                    forget_input_ids,
                    labels=forget_labels,
                    attention_mask=forget_attention_mask,
                )
                idk_logits_oracle = idk_outputs_oracle.logits
                forget_logits_oracle = forget_outputs_oracle.logits

            idk_loss_oracle = -1 * get_batch_loss(idk_logits_oracle, idk_labels)
            forget_loss_oracle = -1 * get_batch_loss(forget_logits_oracle, forget_labels)

            idk_loss_current = -1 * get_batch_loss(idk_outputs.logits, idk_labels)
            forget_loss_current = -1 * get_batch_loss(forget_outputs.logits, forget_labels)

            pi_logratios = idk_loss_current - forget_loss_current
            ref_logratios = idk_loss_oracle - forget_loss_oracle
            #
            beta = 0.1
            loss = -F.logsigmoid(beta * (pi_logratios - ref_logratios)).mean()
            loss = -pi_logratios.mean()
            loss = -idk_loss_current.mean()

            outputs = forget_outputs
        else:
            raise ValueError(f"Invalid loss type {self.loss_type}")

        if (self.l1_lambda is not None and self.l1_lambda != 0) or (self.l0_lambda is not None and self.l0_lambda != 0):
            params = []

            if has_lora_adapter(model):
                # include only the trainable lora parameters
                for param in model.parameters():
                    if param.requires_grad:
                        params.append(param.view(-1))

            else:
                # just calculate the difference with original model
                if self.oracle_model is None:
                    raise ValueError("Oracle model is required for L1\L0 regularization during training without LORA!")
                for param, oracle_param in zip(model.parameters(), self.oracle_model.parameters()):
                    assert param.shape == oracle_param.shape
                    if param.requires_grad:
                        params.append((param - oracle_param.to(model.device)).view(-1))

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


def custom_data_collator_forget(samples):
    # samples: batch_size x (forget, retain, idk(?) ) x (input_ids, label, attention_mask)
    forget, retain, idk = zip(*[(sample[0], sample[1], sample[2] if len(sample) > 2 else None) for sample in samples])

    def stack_data(data):
        input_ids, labels, attention_masks = zip(*data)
        return torch.stack(input_ids), torch.stack(labels), torch.stack(attention_masks)

    forget_input_ids, forget_labels, forget_attention_masks = stack_data(forget)
    retain_input_ids, retain_labels, retain_attention_masks = stack_data(retain)

    rets = [
        [forget_input_ids, forget_labels, forget_attention_masks],
        [retain_input_ids, retain_labels, retain_attention_masks],
    ]
    if any(idk):
        idk_input_ids, idk_labels, idk_attention_masks = stack_data(idk)
        rets.append([idk_input_ids, idk_labels, idk_attention_masks])

    # rets: (forget, retain, idk(?) ) x (input_ids, label, attention_mask) x batch_size
    return rets


def compute_metrics(pred):
    logits, labels = (
        torch.from_numpy(pred.predictions),
        torch.from_numpy(pred.label_ids),
    )
    preds = torch.from_numpy(pred.predictions.argmax(-1))
    shifted_labels = labels[..., 1:].contiguous()
    acc = torch.mean((preds[..., :-1] == shifted_labels).float())
    loss = get_loss(logits, labels)
    return {"eval accuracy": acc, "eval loss": loss.item()}


def get_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    loss = loss_function(output.view(-1, output.size(-1)), shifted_labels.view(-1))

    return loss
