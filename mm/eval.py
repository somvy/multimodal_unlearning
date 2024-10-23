import os
import json
from pathlib import Path
from functools import partial

import evaluate
import hydra
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForPreTraining, AutoConfig
from rouge_score import rouge_scorer

from utils import get_model_identifiers_from_yaml
from trainer_utils import get_batch_loss, remove_image_tokens
from dataset import ImageCaptioningDataset, mm_data_collator_preprocessor
# from eco.main import get_eco_model


def eval_accuracy(logits, labels):
    preds = logits.argmax(-1)
    shifted_labels = labels[..., 1:].contiguous()
    # the places where labels is -100 should be ignored in the accuracy computation
    mask = shifted_labels != -100
    acc = (preds[..., :-1] == shifted_labels).float()
    acc *= mask.float()
    acc = acc.sum() / mask.float().sum()

    return {"eval accuracy": acc.item()}


def eval_rouge_recall(gen_outputs, ground_truths, indices):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    rouge1_recall = {}
    rougeL_recall = {}
    for gen, gt, idx in zip(gen_outputs, ground_truths, indices):
        rouge_scores = scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores["rouge1"].recall
        rougeL_recall[idx] = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def eval_bleu(gen_outputs, ground_truths):
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    rouge_res = rouge.compute(predictions=gen_outputs, references=ground_truths)
    bleu_res = bleu.compute(predictions=gen_outputs, references=ground_truths)

    eval_result = {
        "rouge": rouge_res,
        "bleu": bleu_res,
    }
    return eval_result


def eval_perturbation_ratio(eval_dataloader, perturb_dataloader, model, processor, cfg):
    eval_logs = {}
    for batch, perturb_batch in tqdm(zip(eval_dataloader, perturb_dataloader)):
        # perturb_input_ids, perturb_labels, perturb_attention_mask, _ = perturb_batch
        indices = batch.pop("indices")
        perturb_indices = perturb_batch.pop("indices")

        if len(perturb_batch["input_ids"].shape) > 2:
            bsz, seq_len = perturb_batch["input_ids"].shape[0:2]
            base_shapes = {k: v[0][0].shape for k, v in perturb_batch.items()}
            perturb_batch = {k: v.view(bsz * seq_len, *base_shapes[k]) for k, v in perturb_batch.items()}
        else:
            bsz = perturb_batch["input_ids"].shape[0]
            seq_len = 1

        # send to device
        for k in batch.keys():
            batch[k] = batch[k].to(model.device)
            perturb_batch[k] = perturb_batch[k].to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            perturb_outputs = model(**perturb_batch)

        out_logits = remove_image_tokens(batch["input_ids"], outputs.logits, model.config.image_token_index)
        gt_loss = get_batch_loss(out_logits, batch["labels"])

        perturb_outputs = remove_image_tokens(
            perturb_batch["input_ids"],
            perturb_outputs.logits,
            model.config.image_token_index,
        )
        perturb_loss = get_batch_loss(perturb_outputs, perturb_batch["labels"]).view(bsz, seq_len)

        num_token_gt = (batch["labels"] != -100).sum(-1)
        num_token_perturb = (perturb_batch["labels"] != -100).view(bsz, seq_len, -1).sum(-1)

        mean_perturb_loss = perturb_loss.mean(dim=1)

        ratio = (mean_perturb_loss - gt_loss).mean()

        # eval_logs["perplexity delta"] = eval_logs.get("perplexity delta", []) + [ratio.item()]

        # eval_logs['ground_truth_loss'] = eval_logs.get('ground_truth_loss', []) + [gt_loss.mean().item()]
        # eval_logs['perturb_loss'] = eval_logs.get('perturb_loss', []) + [mean_perturb_loss.mean().item()]

        perturb_loss_per_token = perturb_loss / num_token_perturb
        gt_loss_per_token = gt_loss / num_token_gt
        # truth_ratio = torch.exp(-1 * perturb_loss_per_token).mean(-1) / torch.exp(-1 * gt_loss_per_token)
        truth_ratio = torch.exp(gt_loss_per_token - perturb_loss_per_token.mean(-1))
        # zip index and each stat into a dict
        perturb_loss_per_token = dict(
            zip(
                indices.cpu().numpy().tolist(),
                perturb_loss_per_token.cpu().numpy().tolist(),
            )
        )
        gt_loss_per_token = dict(zip(indices.cpu().numpy().tolist(), gt_loss_per_token.cpu().numpy().tolist()))
        truth_ratio = dict(zip(indices.cpu().numpy().tolist(), truth_ratio.cpu().numpy().tolist()))
        gt_loss = dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist()))
        perturb_loss = dict(zip(indices.cpu().numpy().tolist(), perturb_loss.cpu().numpy().tolist()))
        num_token_gt = dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist()))
        num_token_perturb = dict(zip(indices.cpu().numpy().tolist(), num_token_perturb.cpu().numpy().tolist()))

        # merge dicts

        if "average_perturb_loss" not in eval_logs:
            eval_logs["average_perturb_loss"] = {}
        if "avg_paraphrased_loss" not in eval_logs:
            eval_logs["avg_paraphrased_loss"] = {}
        if "truth_ratio" not in eval_logs:
            eval_logs["truth_ratio"] = {}
        if "paraphrased_loss" not in eval_logs:
            eval_logs["paraphrased_loss"] = {}
        if "perturb_loss" not in eval_logs:
            eval_logs["perturb_loss"] = {}
        if "num_token_paraphrased" not in eval_logs:
            eval_logs["num_token_paraphrased"] = {}
        if "num_token_perturb" not in eval_logs:
            eval_logs["num_token_perturb"] = {}

        eval_logs["average_perturb_loss"].update(perturb_loss_per_token)
        eval_logs["avg_paraphrased_loss"].update(gt_loss_per_token)
        eval_logs["truth_ratio"].update(truth_ratio)
        eval_logs["paraphrased_loss"].update(gt_loss)
        eval_logs["perturb_loss"].update(perturb_loss)
        eval_logs["num_token_paraphrased"].update(num_token_gt)
        eval_logs["num_token_perturb"].update(num_token_perturb)

    return eval_logs


def run_generation(cfg, batch, model, proressor):
    input_ids = batch["input_ids"]
    input_strings = proressor.batch_decode(input_ids, skip_special_tokens=True)
    # print(input_strings)
    split_symbol = "ASSISTANT: "
    ground_truth = [s.split(split_symbol)[1] for s in input_strings]
    input_strings = [s.split(split_symbol)[0] for s in input_strings]
    # add ["/INST "] to the end of each string
    if cfg.model_family.startswith("llama2-7b"):
        input_strings = [s + split_symbol for s in input_strings]

    # we only want to retain the input before the [/INST] token. split each string to only retain the content before the [/INST] token
    # ground_truth = [s.split("[/INST] ")[1] for s in input_strings]
    # input_strings = [s.split("[/INST] ")[0] for s in input_strings]
    # #add ["/INST "] to the end of each string
    # input_strings = [s + "[/INST] " for s in input_strings]

    # now tokenize the strings with left padding
    left_pad_tokenizer = proressor
    left_pad_tokenizer.tokenizer.padding_side = "left"
    left_pad_tokenizer.tokenizer.padding_size = "longest"
    left_pad_tokenizer.tokenizer.pad_token = left_pad_tokenizer.tokenizer.eos_token
    left_pad_tokenizer.tokenizer.pad_token_id = left_pad_tokenizer.tokenizer.eos_token_id

    inputs = left_pad_tokenizer.tokenizer.batch_encode_plus(
        input_strings, add_special_tokens=True, return_tensors="pt", padding=True
    ).to(model.device)
    # print("####################")
    # print("prompts: ", input_strings)
    generate_kwargs = dict(
        **inputs,
        max_new_tokens=cfg.generation.max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=left_pad_tokenizer.tokenizer.eos_token_id,
    )
    out = model.generate(**generate_kwargs)
    strs = left_pad_tokenizer.batch_decode(out[:, inputs.input_ids.shape[-1] :], skip_special_tokens=True)
    return input_strings, strs, ground_truth


def get_dataloader(
    question_strategy,
    question_key,
    cap_key,
    folder,
    split,
    batch_size,
    ds_size,
    data_collator,
):
    ds = ImageCaptioningDataset(
        folder,
        split=split,
        caption_key=cap_key,
        question_strategy=question_strategy,
        question_key=question_key,
    )
    if ds_size:
        ds.data = ds.data.select(range(min(ds_size, len(ds.data))))
    return DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)


def get_all_evals(
    cfg,
    model,
    processor,
    eval_task,
    eval_dl,
    base_eval_dl,
    perturb_dl,
    normalize_gt=False,
):
    eval_logs = {}

    gen_outputs = []
    ground_truths = []
    input_strings = []
    all_indices = []

    eval_logs.update(eval_perturbation_ratio(base_eval_dl, perturb_dl, model, processor, cfg))

    for batch in tqdm(eval_dl):
        indices = batch.pop("indices")

        for k, v in batch.items():
            batch[k] = v.to(model.device)

        with torch.no_grad():
            outputs = model(**batch)
            input_string, gen_output, gt = run_generation(cfg, batch, model, processor)
            gen_outputs.extend(gen_output)
            ground_truths.extend(gt)
            input_strings.extend(input_string)
        # print(outputs)
        out_logits = remove_image_tokens(batch["input_ids"], outputs.logits, model.config.image_token_index)
        # print(out_logits.shape, batch["labels"].shape)
        gt_loss = get_batch_loss(out_logits, batch["labels"])
        num_token_gt = (batch["labels"] != -100).sum(-1)
        gt_loss_per_token = gt_loss / num_token_gt

        if "avg_gt_loss" not in eval_logs:
            eval_logs["avg_gt_loss"] = {}
        if "gt_loss" not in eval_logs:
            eval_logs["gt_loss"] = {}
        if "num_token_gt" not in eval_logs:
            eval_logs["num_token_gt"] = {}
        if "generated_text" not in eval_logs:
            eval_logs["generated_text"] = {}
        # print(gt_loss.shape, num_token_gt.shape)
        eval_logs["avg_gt_loss"].update(
            dict(
                zip(
                    indices.cpu().numpy().tolist(),
                    gt_loss_per_token.cpu().numpy().tolist(),
                )
            )
        )
        eval_logs["gt_loss"].update(dict(zip(indices.cpu().numpy().tolist(), gt_loss.cpu().numpy().tolist())))
        eval_logs["num_token_gt"].update(dict(zip(indices.cpu().numpy().tolist(), num_token_gt.cpu().numpy().tolist())))
        eval_logs["generated_text"].update(dict(zip(indices.cpu().numpy().tolist(), zip(input_string, gen_output, gt))))

    eval_logs.update(eval_rouge_recall(gen_outputs, ground_truths, all_indices))

    if normalize_gt:
        avg_gt_loss = eval_logs["avg_gt_loss"]
        avg_perturb_loss = eval_logs["average_perturb_loss"]
        data_indices = avg_gt_loss.keys()
        normalized_gt_loss = {}
        for idx in data_indices:
            truth_prob = np.exp(-1 * avg_gt_loss[idx])
            perturb_prob = np.exp(-1 * np.array(avg_perturb_loss[idx]))
            all_prob = np.array([truth_prob, *perturb_prob])
            normalized_gt_prob = truth_prob / all_prob.sum()
            normalized_gt_loss[idx] = -1 * np.log(normalized_gt_prob)

        eval_logs["normalized_gt_loss"] = normalized_gt_loss

    return eval_logs


@hydra.main(version_base=None, config_path="../config/mm", config_name="eval")
def main(cfg):
    assert (
        len(cfg.data_path)
        == len(cfg.split_list)
        == len(cfg.eval_task)
        == len(cfg.question_key)
        == len(cfg.answer_key)
        == len(cfg.base_answer_key)
        == len(cfg.perturbed_answer_key)
    ), "data_path, split, eval_task, question_key, and answer_key must be the same length"
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)

    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
    else:
        device_map = {"": 0}

    os.environ["WANDB_DISABLED"] = "true"
    model_cfg = get_model_identifiers_from_yaml(cfg.model_family)
    model_id = model_cfg["hf_key"]
    processor = AutoProcessor.from_pretrained(cfg.model_path)
    processor.tokenizer.padding_side = "left"
    processor.do_pad = True
    # processor.tokenizer.pad_token = processor.tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_id)
    model = AutoModelForPreTraining.from_pretrained(
        cfg.model_path,
        config=config,
        use_flash_attention_2=model_cfg["flash_attention2"] == "true",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()

    # write custom eval loop using compute_metrics
    aggregated_eval_logs = {}
    for i, (
        folder,
        split,
        question_key,
        question_strategy,
        answer_key,
        eval_task,
        base_answer_key,
        perturbed_answer_key,
    ) in enumerate(
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
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"Working on eval task {eval_task} with split {split}")
        save_filename = os.path.join(cfg.save_dir, f"{eval_task}.json")
        save_filename = (
            save_filename
            if world_size == 1
            else os.path.join(cfg.save_dir, f"{eval_task}_{os.environ.get('LOCAL_RANK', '0')}.json")
        )

        if os.path.exists(save_filename) and not cfg.overwrite:
            print(f"Skipping {eval_task} because {save_filename} already exists")
            continue

        data_collator = partial(
            mm_data_collator_preprocessor,
            processor=processor,
            max_length=cfg.max_length,
            return_indices=True,
        )
        eval_dataloader, base_eval_dataloader, perturb_dataloader = (
            get_dataloader(
                question_strategy,
                question_key,
                cap_key,
                folder,
                split,
                batch_size,
                cfg.ds_size,
                data_collator,
            )
            for cap_key, batch_size in zip(
                (answer_key, base_answer_key, perturbed_answer_key),
                (cfg.batch_size, cfg.batch_size // 4, cfg.batch_size // 4),
            )
        )

        normalize_gt = False
        if "eval_log" not in eval_task:
            normalize_gt = True
        eval_logs = get_all_evals(
            cfg,
            model,
            processor,
            eval_task,
            eval_dataloader,
            base_eval_dataloader,
            perturb_dataloader,
            normalize_gt=normalize_gt,
        )

        with open(save_filename, "w") as f:
            # pretty write json to f
            json.dump(eval_logs, f, indent=4)

        aggregated_eval_logs[f"{eval_task}.json"] = eval_logs

    aggregated_eval_log_filename = os.path.join(cfg.save_dir, "eval_log_aggregated.json")

    with open(aggregated_eval_log_filename, "w") as f:
        # pretty write json to f
        json.dump(aggregated_eval_logs, f, indent=4)


if __name__ == "__main__":
    main()
