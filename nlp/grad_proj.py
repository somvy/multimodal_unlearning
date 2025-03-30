import inspect
import math
import os
import shutil
import sys
import time
from typing import Optional

import torch
from torch import nn
from torch.utils.data import SequentialSampler
from transformers import DataCollatorWithPadding, Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import deepspeed_init
from transformers.trainer_callback import ExportableState, TrainerState
from transformers.trainer_pt_utils import get_model_param_count
from transformers.trainer_utils import HPSearchBackend, TrainOutput, has_length, speed_metrics
from transformers.training_args import OptimizerNames
from transformers.utils import is_accelerate_available, is_sagemaker_mp_enabled, is_torch_xla_available, logging

PROJECTION_METHODS = ("grad_proj", "grad_proj_l2")


if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate.utils import (
        DistributedType,
    )


logger = logging.get_logger(__name__)


class GradProjectionsTrainer(Trainer):
    def __init__(self, **kwargs):
        self.forget_loss = kwargs.pop("forget_loss")
        if self.forget_loss not in ("grad_proj_l2", "grad_proj"):
            raise ValueError(f"Invalid forget loss: {self.forget_loss}")
        self.l2_grad_gamma = kwargs.pop("l2_grad_gamma")
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=True):
        # if "factor" not in inputs.keys():
        #     return super().compute_loss(model, inputs, return_outputs)

        positive_inputs, negative_inputs = inputs

        if len(negative_inputs[0]) != 0:
            # negative_inputs["input_ids"] = negative_inputs["input_ids"].reshape(negative_inputs["input_ids"].shape[0] * negative_inputs["input_ids"].shape[1])
            # print('Negative inputs:')
            # print(negative_inputs)
            negative_input_ids, negative_labels, negative_attn = negative_inputs
            # print(negative_inputs['input_ids'].shape)
            negative_outputs = model(input_ids=negative_input_ids, attention_mask=negative_attn, labels=negative_labels)
            # del negative_inputs
            negative_logits = negative_outputs.logits
            # del negative_outputs
            # torch.cuda.empty_cache()
            negative_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits_neg = negative_logits[..., :-1, :].contiguous()
            shift_labels_neg = negative_labels[..., 1:].contiguous()
            # del negative_logits
            # del negative_labels
            # torch.cuda.empty_cache()
            negative_loss = negative_loss_fct(shift_logits_neg.reshape(-1, shift_logits_neg.size(-1)), shift_labels_neg.reshape(-1))
            valid_counts_neg = (shift_labels_neg != -100).sum(dim=-1).float()
            negative_loss = negative_loss.view(shift_logits_neg.size(0), -1)
            # del shift_logits_neg
            # del shift_labels_neg
            # torch.cuda.empty_cache()
            negative_loss = negative_loss.sum(dim=-1) / valid_counts_neg
            negative_loss = (negative_loss * -1.0).mean()
            nf = True
        else:
            negative_loss = 0.0
            nf = False

        if len(positive_inputs[0]) != 0:
            # positive_inputs["input_ids"] = positive_inputs["input_ids"].reshape(positive_inputs["input_ids"].shape[0] * positive_inputs["input_ids"].shape[1])
            # print('Positive inputs:')
            # print(positive_inputs)
            positive_input_ids, positive_labels, positive_attn = positive_inputs
            positive_outputs = model(input_ids=positive_input_ids, attention_mask=positive_attn, labels=positive_labels)

            # del positive_inputs
            positive_logits = positive_outputs.logits
            # del positive_outputs
            # torch.cuda.empty_cache()
            positive_loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits_pos = positive_logits[..., :-1, :].contiguous()
            # del positive_logits
            shift_labels_pos = positive_labels[..., 1:].contiguous()
            # del positive_labels
            # torch.cuda.empty_cache()
            positive_loss = positive_loss_fct(shift_logits_pos.reshape(-1, shift_logits_pos.size(-1)), shift_labels_pos.reshape(-1))
            valid_counts_pos = (shift_labels_pos != -100).sum(dim=-1).float()
            positive_loss = positive_loss.view(shift_logits_pos.size(0), -1)
            # del shift_logits_pos
            # del shift_labels_pos
            # torch.cuda.empty_cache()
            positive_loss = positive_loss.sum(dim=-1) / valid_counts_pos
            positive_loss = (positive_loss * 1.0).mean()
            pf = True

        else:
            positive_loss = 0.0
            pf = False

        loss = negative_loss + positive_loss
        return (loss, negative_loss, positive_loss, nf, pf) if return_outputs else loss

    def _inner_training_loop(self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(args.max_steps % num_update_steps_per_epoch > 0)
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError("args.max_steps must be set to a positive value if dataloader does not have a length, was" f" {args.max_steps}")

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState(
            stateful_callbacks=[cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)]
        )
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        elif self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            # In this case we are in DDP + LOMO, which should be supported
            self.optimizer = self.accelerator.prepare(self.optimizer)

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model))
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            self._load_callback_state()
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first" f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = len(epoch_iterator) if len_dataloader is not None else args.max_steps * args.gradient_accumulation_steps
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            grads = {}
            grads["neg"] = {}
            grads["pos"] = {}
            nc = 0
            pc = 0
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64))
                        ).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step, grads, nc, pc = self.training_step(model, inputs, grads, nc, pc)

                if args.logging_nan_inf_filter and not is_torch_xla_available() and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step)):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if is_accelerate_available() and self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    # self.optimizer.step()

                    ##############################################################################################
                    # print(grads["neg"])
                    for idx in grads["neg"].keys():
                        grads["neg"][idx] /= nc

                    for idx in grads["pos"].keys():
                        grads["pos"][idx] /= pc

                    if self.forget_loss == "grad_proj":
                        for idx, param in enumerate(model.parameters()):
                            if param.grad is not None and param.requires_grad:
                                pos_grad = grads["pos"][idx].to(param.grad.data.device)
                                neg_grad = grads["neg"][idx].to(param.grad.data.device)
                                inner_product = torch.dot(torch.flatten(neg_grad), torch.flatten(pos_grad))
                                coef = inner_product / torch.norm(pos_grad) ** 2

                                new_grad = neg_grad - min(coef, 0) * pos_grad
                                param.grad.data = new_grad

                    elif self.forget_loss == "grad_proj_l2":
                        new_loss = torch.tensor(0.0, requires_grad=True).to(tr_loss.device)
                        # new_loss = tr_loss
                        for idx, param in enumerate(model.parameters()):
                            if param.grad is not None and param.requires_grad:
                                pos_grad = grads["pos"][idx]  # .to(param.grad.data.device)
                                neg_grad = grads["neg"][idx]  # .to(param.grad.data.device)
                                grad_diff = ((neg_grad - pos_grad) ** 2).sum()

                                # print(neg_grad.grad)
                                new_loss += self.l2_grad_gamma * grad_diff

                        # print(new_loss, self.l2_grad_gamma)
                        new_loss.backward(retain_graph=True)

                    ##############################################################################################
                    self.optimizer.step()

                    # self.control = self.callback_handler.on_optimizer_step(args, self.state, self.control)

                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    grads = {}
                    grads["neg"] = {}
                    grads["pos"] = {}
                    nc = 0
                    pc = 0
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break

            # torch.cuda.empty_cache()
            # del grads

            # self.optimizer.step()

            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def training_step(self, model, inputs, grads, pc, nc):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss, negative_loss, positive_loss, nf, pf = self.compute_loss(model, inputs)

        # del inputs
        # torch.cuda.empty_cache()

        kwargs = {
            "retain_graph": False if self.forget_loss == "grad_proj" else True,
        }

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if nf:
            self.optimizer.zero_grad()
            model.zero_grad()
            # DeepSpeedEngine.backward(loss)
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                self.accelerator.deepspeed_engine.backward(negative_loss, **kwargs)
            elif self.accelerator.scaler is not None:
                self.accelerator.scaler.scale(negative_loss).backward(**kwargs)
            else:
                negative_loss.backward(**kwargs)

            # print(negative_loss.grad)
            nc += 1
            for idx, param in enumerate(model.parameters()):
                if param.grad is not None and param.requires_grad:
                    if self.forget_loss == "grad_proj":
                        if idx not in grads["neg"]:
                            grads["neg"][idx] = param.grad.data.detach().clone().cpu().to(param.grad.data.dtype)
                        else:
                            grads["neg"][idx] += param.grad.data.detach().clone().cpu().to(param.grad.data.dtype)

                    elif self.forget_loss == "grad_proj_l2":
                        if idx not in grads["neg"]:
                            grads["neg"][idx] = param.grad
                        else:
                            grads["neg"][idx] += param.grad
                    else:
                        raise ValueError(f"Invalid forget loss: {self.forget_loss}")
            # torch.cuda.empty_cache()

        if pf:
            self.optimizer.zero_grad()
            model.zero_grad()
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                self.accelerator.deepspeed_engine.backward(positive_loss, **kwargs)
            elif self.accelerator.scaler is not None:
                self.accelerator.scaler.scale(positive_loss).backward(**kwargs)
            else:
                positive_loss.backward(**kwargs)

            pc += 1
            for idx, param in enumerate(model.parameters()):
                if param.grad is not None and param.requires_grad:
                    if self.forget_loss == "grad_proj":
                        if idx not in grads["pos"]:
                            grads["pos"][idx] = param.grad.data.detach().clone().cpu().to(param.grad.data.dtype)
                        else:
                            grads["pos"][idx] += param.grad.data.detach().clone().cpu().to(param.grad.data.dtype)

                    elif self.forget_loss == "grad_proj_l2":
                        if idx not in grads["pos"]:
                            grads["pos"][idx] = param.grad
                        else:
                            grads["pos"][idx] += param.grad
                    else:
                        raise ValueError(f"Invalid forget loss: {self.forget_loss}")

        # print(grads["pos"][2])
        torch.cuda.empty_cache()
        return loss.detach() / self.args.gradient_accumulation_steps, grads, nc, pc

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        return SequentialSampler(self.train_dataset)

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += list(set(["label", "label_ids"] + self.label_names))
            self._signature_columns.append("factor")


class AscentPlusDescentDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        batch = super().__call__(features)
        # print([f["factor"] for f in features])
        if "factor" in features[0].keys():
            batch["factor"] = torch.tensor([f["factor"] for f in features])
        return batch
