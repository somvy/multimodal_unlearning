import json

import hydra
import numpy as np
from rouge_score import rouge_scorer
from scipy.spatial.distance import jensenshannon
from scipy.stats import hmean, ks_2samp

rouge_scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def aggregate_to_file(eval_folder):
    # Aggregates evaluation statistics from JSON files and writes to a new file
    aggregated_data = {}
    files_to_aggregate = (
        "log_forget",
        "real_author_wo_options",
        "real_world_wo_options",
        "log",
    )

    for file in files_to_aggregate:
        file_path = eval_folder / f"eval_{file}.json"
        with open(file_path) as f:
            aggregated_data[file] = json.load(f)

    output_file = eval_folder / "eval_log_aggregated.json"
    with open(output_file, "w") as f:
        json.dump(aggregated_data, f, indent=4)

    return aggregated_data


def eval_rouge_recall(text_pairs: dict):
    rouge1_recall = {}
    rougeL_recall = {}
    for idx, pair in text_pairs.items():
        _, gen, gt, *_ = pair
        rouge_scores = rouge_scorer.score(gt, gen)
        rouge1_recall[idx] = rouge_scores["rouge1"].recall
        rougeL_recall[idx] = rouge_scores["rougeL"].recall

    return {"rouge1_recall": rouge1_recall, "rougeL_recall": rougeL_recall}


def dict_values_to_array(d: dict):
    return np.array(list(d.values()))


# Computes quality of forgetfulness based on statistical comparison
def evaluate_forget_quality(unlearned_data, retained_data):
    # Extract necessary data from the results
    unlearned_forget = unlearned_data["eval_log_forget.json"]
    retained_forget = retained_data["eval_log_forget.json"]

    # Calculate mean paraphrase and perturbation loss for unlearned and retained data
    unlearned_paraphrase_loss = dict_values_to_array(unlearned_forget["avg_paraphrased_loss"])
    unlearned_perturbed_loss = dict_values_to_array(unlearned_forget["average_perturb_loss"]).mean()

    retained_paraphrase_loss = dict_values_to_array(retained_forget["avg_paraphrased_loss"])
    retained_perturbed_loss = dict_values_to_array(retained_forget["average_perturb_loss"]).mean()

    # Compute truth ratio using exponential difference of losses
    unlearned_truth_ratio = np.exp(unlearned_perturbed_loss - unlearned_paraphrase_loss)
    retained_truth_ratio = np.exp(retained_perturbed_loss - retained_paraphrase_loss)

    # Perform KS statistical test to compare distributions

    return {
        "KS test p-value": ks_2samp(unlearned_truth_ratio, retained_truth_ratio).pvalue,
        "JS metric": jensenshannon(unlearned_truth_ratio, retained_truth_ratio),
    }


FILE_TO_TASK = {
    "eval_real_faces_wo_options.json": "Real Faces",
    "eval_real_world_wo_options.json": "Real World",
    "eval_log.json": "Retain",
    "eval_retain_facerec.json": "Retain FaceRec",
    "eval_log_forget.json": "Forget",
    "eval_forget_facerec.json": "Forget FaceRec",
}

METRIC_NAMES = ("ROUGE", "Prob.", "Truth Ratio")


# Computes model utility based on evaluation results
def compute_model_utility(eval_results):
    aggregated_results = {}

    # Iterate through each evaluation task to calculate METRIC_NAMES
    for task_file, task_result in eval_results.items():
        task_name = FILE_TO_TASK[task_file]
        # Probability calculation
        if "eval_log" in task_file:
            true_probs = np.exp(-dict_values_to_array(task_result["avg_gt_loss"]))
            avg_gt_prob = np.mean(true_probs)
        else:
            true_probs = np.exp(-dict_values_to_array(task_result["avg_gt_loss"]))
            false_probs = np.exp(-dict_values_to_array(task_result["average_perturb_loss"]))
            combined_probs = np.concatenate([np.expand_dims(true_probs, axis=-1), false_probs], axis=1).sum(-1)
            avg_gt_prob = np.mean(true_probs / combined_probs)

        aggregated_results[f"Prob. {task_name}"] = avg_gt_prob

        # ROUGE score calculation
        if len(task_result.get("rougeL_recall", [])) == 0:
            task_result.update(eval_rouge_recall(task_result["generated_text"]))

        # calculate average ROUGE score from generated text
        aggregated_results[f"ROUGE {task_name}"] = dict_values_to_array(task_result["rougeL_recall"]).mean()

        # Truth Ratio calculation
        paraphrase_loss = dict_values_to_array(task_result["avg_paraphrased_loss"])
        perturbed_loss = dict_values_to_array(task_result["average_perturb_loss"]).mean()
        truth_ratio = np.exp(perturbed_loss - paraphrase_loss)

        if "forget" in task_file:
            truth_ratio_value = np.mean(np.minimum(truth_ratio, 1 / truth_ratio))
        else:
            truth_ratio_value = np.mean(np.maximum(0, 1 - 1 / truth_ratio))

        aggregated_results[f"Truth Ratio {task_name}"] = truth_ratio_value

    # Calculate harmonic mean of utilities for non-forget tasks
    aggregated_results["Model Utility"] = hmean([v for k, v in aggregated_results.items() if "Forget" not in k])

    return aggregated_results


@hydra.main(version_base=None, config_path="../config/mm", config_name="calculate_metrics")
def main(cfg):
    with open(cfg.retain_result) as f:
        retain_data = json.load(f)

    with open(cfg.ckpt_result) as f:
        checkpoint_data = json.load(f)

    results = dict(
        method=cfg.method_name,
        submitted_by=cfg.submitted_by,
        **compute_model_utility(checkpoint_data),
        **evaluate_forget_quality(checkpoint_data, retain_data),
    )


    with open(cfg.save_file, "w") as f:
        json.dump(results, f, indent=4)

    print(results)
    
if __name__ == "__main__":
    main()
