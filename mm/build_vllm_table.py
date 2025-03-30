import json
import subprocess
from pathlib import Path

import pandas as pd
from scipy.stats import hmean
from tqdm import tqdm

RES_FOLDER = Path("mm_results")
GOLD_MODEL_PATH = Path("/home/dontsov/unlearning/models/llava/ft_retain90+tofu_epoch3_lr1e-05__wd0.01_lora")
BASE_MODEL_PATH = Path("/home/dontsov/unlearning/models/llava/ft_full+tofu_epoch3_lr1e-05__wd0.01_lora")


def run_aggregate_res(model_folder: Path, res_folder, reeval=False):
    eval_results = model_folder / "eval_results" / "eval_log_aggregated.json"
    gold_results = GOLD_MODEL_PATH / "eval_results" / "eval_log_aggregated.json"

    if not eval_results.exists():
        raise FileNotFoundError(f"File {eval_results} not found")

    res_file = res_folder / (model_folder.name + ".json")
    if res_file.exists() and not reeval:
        print(f"Skipping {model_folder}, due to existing results")
        return

    cmd = [
        f'cd ../ && python mm/calculate_metrics.py "retain_result={gold_results.absolute()}" "ckpt_result={eval_results.absolute()}" "method_name={model_folder.name}" "save_file={res_file.absolute()}"'
    ]
    subprocess.run(cmd, shell=True, check=True)


FILE_TO_NAME = {
    "dpo_forget10+tofu_lr1e-05_5": "DPO",
    "grad_diff_forget_ce_retain_ce_forget10_L1_0.0_fromzero": "GD",
    "grad_ascent_forget10+tofu": "GA",
    "idk_forget10+tofu": "IDK",
    "KL_forget10+tofu": "KL",
    "LLMU_forget10+tofu_lr1e-05_5": "LLMU",
    "npo_forget10_L1_0.0_fromzero": "NPO",
    "retain_ft_forget10+tofu": "Retain FT",
    "rmu_forget10_L1_0.0_fromzero": "RMU",
    "scrub_forget10+tofu_lr1e-05_5": "SCRUB",
    "SKU_forget10_L1_0.0_fromzero": "SKU",
}


def main():
    for model_name in tqdm(FILE_TO_NAME.keys()):
        run_aggregate_res(BASE_MODEL_PATH / model_name, RES_FOLDER)

    run_aggregate_res(BASE_MODEL_PATH, RES_FOLDER)
    run_aggregate_res(GOLD_MODEL_PATH, RES_FOLDER)

    base_res = json.loads((RES_FOLDER / (BASE_MODEL_PATH.name + ".json")).read_text())
    base_res["Method"] = "Base"

    gold_res = json.loads((RES_FOLDER / (GOLD_MODEL_PATH.name + ".json")).read_text())
    gold_res["Method"] = "Gold"

    result_csvs = [gold_res, base_res]

    for res_file, method_name in FILE_TO_NAME.items():
        res_path = RES_FOLDER / (res_file + ".json")
        result_csvs.append(
            {
                **json.loads(res_path.read_text()),
                "Method": method_name,
            }
        )

    res_df = pd.DataFrame.from_dict(result_csvs)

    real_metrics = ["Prob. Real Faces", "Truth Ratio Real Faces", "Prob. Real World", "Truth Ratio Real World"]
    retain_metrics = ["Prob. Retain", "Truth Ratio Retain", "Prob. Retain FaceRec", "Truth Ratio Retain FaceRec"]
    forget_metrics = ["Prob. Forget", "Truth Ratio Forget", "Prob. Forget FaceRec", "Truth Ratio Forget FaceRec"]

    util_metrics = [
        "Prob. Real Faces",
        "Truth Ratio Real Faces",
        # "ROUGE Real World",
        "Prob. Real World",
        "Truth Ratio Real World",
        "ROUGE Retain",
        "Prob. Retain",
        "Truth Ratio Retain",
        "Prob. Retain FaceRec",
        "Truth Ratio Retain FaceRec",
        "ROUGE Forget",
        "Prob. Forget",
        "Truth Ratio Forget",
        "Prob. Forget FaceRec",
        "Truth Ratio Forget FaceRec",
    ]

    def calc_metric(keys):
        return hmean([res_df[key] for key in keys], axis=0)

    res_df["Real metric"] = calc_metric(real_metrics)
    res_df["Forget metric"] = calc_metric(forget_metrics)
    res_df["Retain metric"] = calc_metric(retain_metrics)
    res_df["Model utility"] = calc_metric(util_metrics)

    res_df["Forget Quality"] = 1 - res_df["JS metric"]

    print(res_df)
    res_df.to_json("mm_results.json", indent=4)

    papergray_rows = res_df["Retain metric"] < 0.4
    res_df.loc[papergray_rows, "Method"] = res_df.loc[papergray_rows, "Method"].apply(lambda x: "\\rowcolor{papergray} " + x)

    print(
        res_df[["Method", "Real metric", "Retain metric", "Forget metric", "Forget Quality"]].to_latex(header=True, index=False, float_format="%.2f")
    )


if __name__ == "__main__":
    main()
