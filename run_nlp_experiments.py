import subprocess
import os

from pathlib import Path
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--reversed",action="store_true")
parser.add_argument("--re_eval", action="store_true")


args = parser.parse_args()
logs_dir = Path("logs")
current_env = os.environ.copy()

master_port = current_env.get("MASTER_PORT", 29500)

def run_single_exp(loss, l1, l0):
    # Open the file where the output will be saved
    save_name = f"{loss}_L1{l1}_L0{l0}"
    try:
        save_file = logs_dir / (save_name + ".txt")
        # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

        print(f"Running model with loss={loss}, l1={l1}, l0={l0}")

        save_dir = Path(f"models/llama2-7b/ft_full_epoch5_lr1e-05__wd0.01_lora/{save_name}")
        ckpt_path = save_dir / "checkpoint-500"
        if not ckpt_path.exists():
            print(f"Training...")
            with save_file.open("w") as outfile:
                cmd = [f'torchrun --master_port={master_port} nlp/forget.py "forget_loss={loss}" "l1_lambda={l1}" "l0_lambda={l0}" "save_dir={save_dir}"']

                subprocess.run(cmd, shell=True, stdout=outfile, stderr=outfile,env=current_env)
            
        eval_path = ckpt_path / "eval_results" 

        if not (eval_path / "eval_log_aggregated.json" ).exists() or args.re_eval:
            print(f"Evaluating...")
            with save_file.open("a") as outfile:
                cmd = [f'torchrun --master_port={master_port} nlp/evaluate_util.py "model_path={ckpt_path.absolute()}" "model_family=llama2-7b"']
                subprocess.run(cmd, shell=True, stdout=outfile, stderr=outfile, env=current_env)
        print("Done")

    except KeyboardInterrupt:
        print("Experiment interrupted")
        exit()

    except Exception as e:
        print(f"Error running experiment with loss={loss}, l1={l1}, l0={l0}")
        print(e)


# Define the hyperparameters to run

losses = [
    "retain_ft",
    "grad_ascent",
    "grad_diff_forget_ce_retain_ce",
    "grad_diff_forget_entropy_retain_ce",
    "grad_diff_forget_entropy_retain_KL",
    "grad_diff_forget_ce_retain_KL",
    "grad_diff_forget_KL_retain_KL",
    "grad_diff_forget_KL_retain_CE",
    "KL",
    "idk",
    "dpo",
    "npo",
    "scrub",
    "rmu"
]

l_values = [0.01, 0.1, 1.0]


# Run the experiments
if args.reversed:
    losses = losses[::-1]

for loss in losses:
    run_single_exp(loss, 0, 0)
    # run_single_exp(loss, 0.01, 0.01)

    for l1 in l_values:
        run_single_exp(loss, l1, 0)

    # for l0 in l_values:
    #     run_single_exp(loss, 0, l0)

    # for l in l_values:
    #     run_single_exp(loss, l, l)
