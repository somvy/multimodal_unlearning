import json
from pathlib import Path

import pandas as pd
from build_vllm_table import run_aggregate_res
from scipy.stats import hmean
from tqdm import tqdm

RES_FOLDER = Path("modailties_results")
RES_FOLDER.mkdir(exist_ok=True)

GOLD_MODEL_PATH = Path("/home/dontsov/unlearning/models/llava/ft_retain90+tofu_epoch3_lr1e-05__wd0.01_lora")
BASE_MODEL_PATH = Path("/home/dontsov/unlearning/models/llava/ft_full+tofu_epoch3_lr1e-05__wd0.01_lora")


split_to_name = {"_locuslab_TOFU": "text", "_therem_faces_v1": "visual", "+tofu_therem_faces_v1": "both"}

losses = ["RMU", "grad_diff_forget_ce_retain_ce", "retain_ft", "grad_ascent", "KL", "idk", "npo", "scrub", "dpo", "LLMU"]

for loss in tqdm(losses):
    for split, name in split_to_name.items():
        model_name = f"{loss}_forget10{split}"
        run_aggregate_res(BASE_MODEL_PATH / model_name, RES_FOLDER)

run_aggregate_res(BASE_MODEL_PATH, RES_FOLDER)
run_aggregate_res(GOLD_MODEL_PATH, RES_FOLDER)

base_res = json.loads((RES_FOLDER / (BASE_MODEL_PATH.name + ".json")).read_text())
base_res["Method"] = "Base"

gold_res = json.loads((RES_FOLDER / (GOLD_MODEL_PATH.name + ".json")).read_text())
gold_res["Method"] = "Gold"


result_csvs = [gold_res, base_res]

for loss in losses:
    for split, name in split_to_name.items():
        model_name = f"{loss}_forget10{split}"
        res_file = model_name
        res_path = RES_FOLDER / (res_file + ".json")
        result_csvs.append(
            {
                **json.loads(res_path.read_text()),
                "Method": loss,
                "Modality": name,
            }
        )


df = pd.DataFrame(result_csvs)


real_metrics = ["Prob. Real Faces", "Truth Ratio Real Faces", "Prob. Real World", "Truth Ratio Real World"]
retain_metrics = ["Prob. Retain", "Truth Ratio Retain", "Prob. Retain FaceRec", "Truth Ratio Retain FaceRec"]
forget_metrics = ["Prob. Forget", "Truth Ratio Forget", "Prob. Forget FaceRec", "Truth Ratio Forget FaceRec"]


def calc_metric(keys):
    return hmean([df[key] for key in keys], axis=0)


df["Real metric"] = calc_metric(real_metrics)
df["Forget metric"] = calc_metric(forget_metrics)
df["Retain metric"] = calc_metric(retain_metrics)


df["Forget Quality"] = 1 - df["JS metric"]


df.to_json("modalities_results.json", indent=4)

# papergray_rows = df["Retain metric"] < 0.4
# res_df.loc[papergray_rows, "Method"] = res_df.loc[papergray_rows, "Method"].apply(lambda x: "\\rowcolor{papergray} " + x)

print(
    df[["Method", "Modality", "Real metric", "Retain metric", "Forget metric", "Forget Quality"]].to_latex(
        header=True, index=False, float_format="%.2f"
    )
)

# print(df)
