import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import hydra
import numpy as np
import torch
import transformers
from dataset import ImageCaptioningDataset, mm_data_collator_preprocessor
from rouge_score.rouge_scorer import RougeScorer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from trainer_utils import get_batch_loss, remove_image_tokens
from transformers import AutoConfig, AutoProcessor

from utils import get_model_identifiers_from_yaml

rouge_scorer = RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def eval_accuracy(logits, labels):
    preds = logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = shifted_labels != -100
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def eval_rouge_recall(gen_answers, true_ans, indices):
    rouge1_recall = {}
    rougeL_recall = {}
    for idx, gen, gt in zip(indices, gen_answers, true_ans):
        rouge_scores = rouge_scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores["rouge1"].recall
        rougeL_recall[idx] = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model, processor, cfg):
    eval_logs = defaultdict(dict)

    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        # perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        indices = batch.pop("indices")
        perturb_batch.pop("indices")

        bsz, num_seq = perturb_batch["input_ids"].shape[0:2]
        # Flatten the batch size and sequence length dimensions
        # Process each key according to its specific shape requirements
        for k, v in perturb_batch.items():
            if v.shape[:2] == (bsz, num_seq):
                perturb_batch[k] = v.view(bsz * num_seq, -1)
        for k in batch.keys():
            batch[k] = batch[k].to(model.device)
            perturb_batch[k] = perturb_batch[k].to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        out_logits = remove_image_tokens(batch["input_ids"], outputs.logits, model.config.image_token_id)
        gt_loss = get_batch_loss(out_logits, batch["labels"])

        perturb_outputs = remove_image_tokens(perturb_batch["input_ids"], perturb_outputs.logits, model.config.image_token_id)
        perturb_loss = get_batch_loss(perturb_outputs, perturb_batch["labels"]).view(bsz, num_seq)

        num_token_gt = (batch["labels"] != -100).sum(-1)
        num_token_perturb = (perturb_batch["labels"] != -100).view(bsz, num_seq, -1).sum(-1)

        perturb_loss_per_token = perturb_loss / num_token_perturb
        gt_loss_per_token = gt_loss / num_token_gt

        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))

        # zip index and each stat into a dict
        idx_list = indices.tolist()
        eval_logs["average_perturb_loss"].update(dict(zip(idx_list, perturb_loss_per_token.tolist())))
        eval_logs["avg_paraphrased_loss"].update(dict(zip(idx_list, gt_loss_per_token.tolist())))
        eval_logs["truth_ratio"].update(dict(zip(idx_list, truth_ratio.tolist())))
        eval_logs["paraphrased_loss"].update(dict(zip(idx_list, gt_loss.tolist())))
        eval_logs["perturb_loss"].update(dict(zip(idx_list, perturb_loss.tolist())))
        eval_logs["num_token_paraphrased"].update(dict(zip(idx_list, num_token_gt.tolist())))
        eval_logs["num_token_perturb"].update(dict(zip(idx_list, num_token_perturb.tolist())))

    return eval_logs


def get_dataloader(quest_strat, quest_key, cap_key, fold, split, bs, ds_size, collator):
    ds = ImageCaptioningDataset(fold, split=split, caption_key=cap_key, question_strategy=quest_strat, question_key=quest_key)
    if ds_size:
        ds.data = ds.data.select(range(min(ds_size, len(ds.data))))
    return DataLoader(ds, batch_size=bs, collate_fn=collator)


def get_all_evals(cfg, model, processor, eval_task, eval_dl, base_eval_dl, perturb_dl):
    eval_logs = defaultdict(dict)
    gen_answers, true_ans, all_questions, all_indices = [], [], [], []

    eval_logs.update(eval_perturbation_ratio(base_eval_dl, perturb_dl, model, processor, cfg))

    for batch in tqdm(eval_dl):
        indices = batch.pop("indices").tolist()
        answers: list[str] = processor.batch_decode(batch.pop("answers"), skip_special_tokens=True)

        # first get logits on the original input with answers
        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            logits = model(**batch).logits

        # then get model generation on the question
        str_inputs = processor.batch_decode(batch["input_ids"], skip_special_tokens=True)

        # remove the answers from the input
        questions = [s[: s.rfind(ans)] for s, ans in zip(str_inputs, answers)]
        quest_inputs = processor.tokenizer.batch_encode_plus(questions, add_special_tokens=True, return_tensors="pt", padding=True).to(model.device)

        out = model.generate(
            **quest_inputs,
            max_new_tokens=cfg.generation.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
        gen_output = processor.batch_decode(out[:, quest_inputs.input_ids.shape[-1] :], skip_special_tokens=True)

        gen_answers.extend(gen_output)
        true_ans.extend(answers)
        all_questions.extend(questions)

        out_logits = remove_image_tokens(batch["input_ids"], logits, model.config.image_token_id)
        gt_loss = get_batch_loss(out_logits, batch["labels"])
        num_token_gt = (batch["labels"] != -100).sum(-1)
        gt_loss_per_token = gt_loss / num_token_gt

        eval_logs["avg_gt_loss"].update(dict(zip(indices, gt_loss_per_token.tolist())))
        eval_logs["gt_loss"].update(dict(zip(indices, gt_loss.tolist())))
        eval_logs["num_token_gt"].update(dict(zip(indices, num_token_gt.tolist())))
        eval_logs["generated_text"].update(dict(zip(indices, zip(questions, gen_output, answers))))

    eval_logs.update(eval_rouge_recall(gen_answers, true_ans, all_indices))

    # normalise
    if "eval_log" not in eval_task:
        avg_gt_loss = eval_logs["avg_gt_loss"]
        normalized_gt_loss = {}

        for idx, gt_loss in avg_gt_loss.items():
            truth_prob = np.exp(-1 * gt_loss)
            perturb_prob = np.exp(-1 * np.array(eval_logs["average_perturb_loss"][idx]))
            normalized_gt_loss[idx] = -1 * np.log(truth_prob / (perturb_prob.sum() + truth_prob))

        eval_logs["normalized_gt_loss"] = normalized_gt_loss

    return eval_logs


@hydra.main(version_base=None, config_path="../config/mm", config_name="eval")
def main(cfg):
    if not (
        len(cfg.data_path)
        == len(cfg.split_list)
        == len(cfg.eval_task)
        == len(cfg.question_key)
        == len(cfg.answer_key)
        == len(cfg.base_answer_key)
        == len(cfg.perturbed_answer_key)
    ):
        raise ValueError("data_path, split, eval_task, question_key, and answer_key must be the same length")
    # list of task indices to evaluate
    eval_task_ids = cfg.eval_task_ids if cfg.eval_task_ids != "None" else list(range(len(cfg.data_path)))

    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]

    processor = AutoProcessor.from_pretrained(cfg.processor_path)
    processor.tokenizer.padding_side = "left"
    processor.tokenizer.padding_size = "longest"
    processor.do_pad = True

    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_id)
    model = (
        getattr(transformers, model_cfg["hf_class"])
        .from_pretrained(
            cfg.model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .eval()
        .cuda()
    )
    # iterate through the tasks
    aggregated_eval_logs = {}
    for task_idx, (fol, split, quest_key, quest_strat, ans_key, eval_task, base_ans_key, perturbed_ans_key) in enumerate(
        zip(
            cfg.data_path,
            cfg.split_list,
            cfg.question_key,
            cfg.question_strategy,
            cfg.answer_key,
            cfg.eval_task,
            cfg.base_answer_key,
            cfg.perturbed_answer_key,
        )
    ):
        if task_idx not in eval_task_ids:
            continue

        print(f"Working on eval task {eval_task} with split {split}")
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        collator_with_ans = partial(
            mm_data_collator_preprocessor, processor=processor, max_length=cfg.max_length, return_indices=True, return_answers=True
        )
        eval_dl = get_dataloader(quest_strat, quest_key, ans_key, fol, split, cfg.batch_size, cfg.ds_size, collator_with_ans)

        collator = partial(mm_data_collator_preprocessor, processor=processor, max_length=cfg.max_length, return_indices=True)
        base_dl = get_dataloader(quest_strat, quest_key, base_ans_key, fol, split, cfg.batch_size // 4, cfg.ds_size, collator)
        perturb_dl = get_dataloader(quest_strat, quest_key, perturbed_ans_key, fol, split, cfg.batch_size // 4, cfg.ds_size, collator)

        eval_logs = get_all_evals(cfg, model, processor, eval_task, eval_dl, base_dl, perturb_dl)

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

        # clear cuda cache
        torch.cuda.empty_cache()

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")

    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)


if __name__ == "__main__":
    main()
