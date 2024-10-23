import subprocess
import os
from pathlib import Path
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
import queue

parser = ArgumentParser()
parser.add_argument("--re_eval", action="store_true")
args = parser.parse_args()


EXPERIMENT_NAME = "L1_role"
# EXPERIMENT_NAME = "forget_modalities_new"


logs_dir = Path("logs") / EXPERIMENT_NAME
logs_dir.mkdir(exist_ok=True)

current_env = os.environ.copy()
current_env["WANDB_DISABLED"] = "true"

master_port = current_env.get("MASTER_PORT", 29501)

gpu_list = [0,1,2]
worker_envs = {}
for i, gpu_id in enumerate(gpu_list):
    worker_envs[gpu_id] = current_env.copy()
    worker_envs[gpu_id]["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def task_args_to_str(task: dict):
    return " ".join(f'"{k}={v}"' for k, v in task.items())


def run_single_exp(task: dict, gpu_id: int):
    
    save_name = "_".join(task.values()).replace("/", "_")
    
    print(f"{gpu_id}: Running task {save_name}")
    worker_env = worker_envs[gpu_id]
    MASTER_PORT = int(master_port) + gpu_id
    model_dir = Path(
        f"models/llava/ft_full+tofu_epoch3_lr1e-05__wd0.01_lora/{save_name}"
    )

    try:
        if not (model_dir / "adapter_model.safetensors").exists():
            print("Forgetting...")
            train_task = task.copy()
            train_task["save_dir"] = model_dir.absolute()

            with (logs_dir / f"{save_name}_forget.txt").open("w") as outfile:
                cmd = f"torchrun --master_port={MASTER_PORT} mm/forget.py {task_args_to_str(train_task)}"
                print(cmd)
                subprocess.run(
                    cmd, shell=True, stdout=outfile, stderr=outfile, env=worker_env, check=True
                )

        eval_path = model_dir / "eval_results"

        if not (eval_path / "eval_log_aggregated.json").exists() or args.re_eval:
            print("Evaluating...")
            with (logs_dir / f"{save_name}_eval.txt").open("w") as outfile:
                cmd = f'torchrun --master_port={MASTER_PORT} mm/eval.py "model_path={model_dir.absolute()}"'
                subprocess.run(
                    cmd, shell=True, stdout=outfile, stderr=outfile, env=worker_env, check=True
                )
        print("Done")

    except KeyboardInterrupt:
        print("Experiment interrupted")
        exit()

    except Exception as e:
        print(f"Error running experiment with model {save_name}")
        print(e)


models = [
    "LLMU_forget10_text_lr1e-05_5",
    "LLMU_forget10_visual_lr1e-05_5",
    "LLMU_forget10+tofu_lr1e-05_5",
    "scrub_forget10_text_lr1e-05_5",
    "scrub_forget10_visual_lr1e-05_5",
    "scrub_forget10+tofu_lr1e-05_5",
    "dpo_forget10_text_lr1e-05_5",
    "dpo_forget10+tofu_lr1e-05_5",
    "dpo_forget10_visual_lr1e-05_5",
]

# npo - negative preference optimization
losses = [
    # "scrub",
    # "dpo",
    # "LLMU",
    "RMU",
    "retain_ft",
    "grad_ascent", 
    "KL", 
    "idk", 
    "npo",
    "grad_diff_forget_ce_retain_ce"
]

task_queue = queue.Queue()

for loss in losses:
    # task_queue.put(
    #     {
    #         "forget_loss": loss,
    #         # "forget_split": "forget10",
    #         "l_norm_from": "zero",
    #         "l1_lambda": "0.01",
    #     }
    # )
    task_queue.put(
        {
            "forget_loss": loss,
            "l_norm_from": "init",
            "l1_lambda": "0.01",
            # "forget_split": "forget10+tofu",
        }
    )
    task_queue.put(
        {
            "forget_loss": loss,
            "l1_lambda": "0.0"
            # "forget_split": "forget10",
            # "forget_data_path": "locuslab/TOFU",
        }
    )


def worker(gpu_id):
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
            run_single_exp(task, gpu_id)
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error running task on GPU {gpu_id}")
            print(e)
        finally:
            task_queue.task_done()


print(task_queue.qsize())

with ThreadPoolExecutor(max_workers=len(gpu_list)) as executor:
    futures = [executor.submit(worker, gpu_id) for gpu_id in gpu_list]
    # Wait for all tasks to finish
task_queue.join()
