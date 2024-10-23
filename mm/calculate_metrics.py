import csv
import json
from functools import partial

from pathlib import Path
import evaluate
import hydra
import numpy as np
from omegaconf import OmegaConf
from rouge_score import rouge_scorer
from scipy.stats import hmean, ks_2samp, sem

chrf_compute = partial(evaluate.load("chrf").compute, word_order=2, lowercase=True)
rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


# Aggregates evaluation statistics from JSON files and writes to a new file
def aggregate_statistics(eval_folder):
    aggregated_data = {}
    files_to_aggregate = [
        "log_forget",
        "real_author_wo_options",
        "real_world_wo_options",
        "log",
    ]

    for file in files_to_aggregate:
        file_path = eval_folder / f"eval_{file}.json"
        with open(file_path) as f:
            aggregated_data[file] = json.load(f)

    output_file = eval_folder / "eval_log_aggregated.json"
    with open(output_file, "w") as f:
        json.dump(aggregated_data, f, indent=4)

    return aggregated_data


def eval_chrf(text_pairs: dict):
    chrf_scores = {}
    for idx, pair in text_pairs.items():
        chrf_scores[idx] = chrf_compute(predictions=[pair[1]], references=[[pair[2]]])[
            "score"
        ]

    return {"chrf_scores": chrf_scores}


def eval_rouge_recall(text_pairs: dict):
    rouge1_recall = {}
    rougeL_recall = {}
    for idx, pair in text_pairs.items():
        gen = pair[1]
        gt = pair[2]
        rouge_scores = rouge_scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores["rouge1"].recall
        rougeL_recall[idx] = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


# Computes quality of forgetfulness based on statistical comparison
def evaluate_forget_quality(unlearned_data, retained_data):
    # Extract necessary data from the results
    unlearned_forget = unlearned_data["eval_log_forget.json"]
    retained_forget = retained_data["eval_log_forget.json"]

    # Calculate mean paraphrase and perturbation loss for unlearned and retained data
    unlearned_paraphrase_loss = np.array(
        list(unlearned_forget["avg_paraphrased_loss"].values())
    )
    unlearned_perturbed_loss = np.array(
        list(unlearned_forget["average_perturb_loss"].values())
    ).mean()

    retained_paraphrase_loss = np.array(
        list(retained_forget["avg_paraphrased_loss"].values())
    )
    retained_perturbed_loss = np.array(
        list(retained_forget["average_perturb_loss"].values())
    ).mean()

    # Compute truth ratio using exponential difference of losses
    unlearned_truth_ratio = np.exp(unlearned_perturbed_loss - unlearned_paraphrase_loss)
    retained_truth_ratio = np.exp(retained_perturbed_loss - retained_paraphrase_loss)

    # Perform KS statistical test to compare distributions
    ks_test_result = ks_2samp(unlearned_truth_ratio, retained_truth_ratio)

    return {
        "Forget Quality": ks_test_result.pvalue,
        "KS Test P-Value": ks_test_result.pvalue,
        "KS Test Statistic": ks_test_result.statistic,
    }


# Computes model utility based on evaluation results
def compute_model_utility(eval_results):
    task_labels = {
        "eval_real_faces_wo_options.json": "Real Faces",
        "eval_real_world_wo_options.json": "Real World",
        "eval_log.json": "Retain",
        "eval_retain_facerec.json": "Retain FaceRec",
        "eval_log_forget.json": "Forget",
        "eval_forget_facerec.json": "Forget FaceRec",
    }

    metrics = ["ROUGE", "chrf++", "Prob.", "Truth Ratio"]
    aggregated_results = {}

    # Initialize output structure for storing results per task and metric
    for task_file, task_label in task_labels.items():
        for metric in metrics:
            aggregated_results[f"{metric} {task_label}"] = []

    # Iterate through each evaluation task to calculate metrics
    for task_file, task_result in eval_results.items():
        # Probability calculation
        if "eval_log" in task_file:
            true_probs = np.exp(-np.array(list(task_result["avg_gt_loss"].values())))
            avg_gt_prob = np.mean(true_probs)
        else:
            true_probs = np.exp(-np.array(list(task_result["avg_gt_loss"].values())))
            false_probs = np.exp(
                -np.array(list(task_result["average_perturb_loss"].values()))
            )
            combined_probs = np.concatenate(
                [np.expand_dims(true_probs, axis=-1), false_probs], axis=1
            ).sum(-1)
            avg_gt_prob = np.mean(true_probs / combined_probs)

        aggregated_results[f"Prob. {task_labels[task_file]}"] = avg_gt_prob

        # ROUGE score calculation
        if len(task_result.get("rougeL_recall", [])) == 0:
            # calculate average ROUGE score from generated text
            task_result.update(eval_rouge_recall(task_result["generated_text"]))

        avg_rouge_score = np.mean(np.array(list(task_result["rougeL_recall"].values())))
        aggregated_results[f"ROUGE {task_labels[task_file]}"] = avg_rouge_score

        # CHRF++ score calculation
        if len(task_result.get("chrf_scores", [])) == 0:
            # calculate average CHRF++ score from generated text
            task_result.update(eval_chrf(task_result["generated_text"]))

        avg_chrf_score = np.mean(np.array(list(task_result["chrf_scores"].values())))
        aggregated_results[f"chrf++ {task_labels[task_file]}"] = avg_chrf_score

        # Truth Ratio calculation
        paraphrase_loss = np.array(list(task_result["avg_paraphrased_loss"].values()))
        perturbed_loss = np.array(
            list(task_result["average_perturb_loss"].values())
        ).mean()

        truth_ratio = np.exp(perturbed_loss - paraphrase_loss)

        if "forget" in task_file:
            truth_ratio_value = np.mean(np.minimum(truth_ratio, 1 / truth_ratio))
        else:
            truth_ratio_value = np.mean(np.maximum(0, 1 - 1 / truth_ratio))

        aggregated_results[f"Truth Ratio {task_labels[task_file]}"] = truth_ratio_value

    # Calculate harmonic mean of utilities for non-forget tasks
    model_utility_values = [
        v for k, v in aggregated_results.items() if "Forget" not in k
    ]
    aggregated_results["Model Utility"] = hmean(model_utility_values)

    return aggregated_results


@hydra.main(
    version_base=None, config_path="../config/mm", config_name="calculate_metrics"
)
def main(cfg):
    retain_data = json.load(open(cfg.retain_result))
    checkpoint_data = json.load(open(cfg.ckpt_result))

    model_utility = compute_model_utility(checkpoint_data)
    forget_quality = evaluate_forget_quality(checkpoint_data, retain_data)

    model_utility["Forget Quality"] = forget_quality["Forget Quality"]
    model_utility["Method"] = cfg.method_name
    model_utility["Submitted By"] = cfg.submitted_by

    output_file = cfg.save_file
    with open(output_file, "w") as f:
        writer = csv.DictWriter(f, model_utility.keys())
        writer.writeheader()
        writer.writerow(model_utility)

    return model_utility


if __name__ == "__main__":
    main()
