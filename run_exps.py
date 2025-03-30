import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

EXPERIMENT_NAME = "qwen-vl2"
GPU_LIST = [1]
LOGS_DIR = Path("logs") / EXPERIMENT_NAME
LOGS_DIR.mkdir(exist_ok=True)
BASE_MODEL_PATH = Path("/home/dontsov/unlearning/models/qwen-vl2-2b/ft_full+tofu")

current_env = os.environ.copy()
current_env["WANDB_DISABLED"] = "true"


def task_args_to_str(task: dict):
    return " ".join(f'"{k}={v}"' for k, v in task.items())


def run_single_exp(task: dict, gpu_id: int):
    save_name = "_".join(task.values()).replace("/", "_")

    print(f"{gpu_id}: Running task {save_name}")

    model_dir = BASE_MODEL_PATH / save_name
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not (model_dir / "adapter_model.safetensors").exists():
            print("Forgetting...")
            train_task = task.copy()
            train_task["save_dir"] = model_dir.absolute()

            with (LOGS_DIR / f"{save_name}_forget.txt").open("w") as outfile:
                cmd = f"accelerate launch --gpu-ids {gpu_id} mm/forget.py {task_args_to_str(train_task)}"
                print(cmd)
                subprocess.run(cmd, shell=True, stdout=outfile, stderr=outfile, env=current_env, check=True)

        eval_path = model_dir / "eval_results"

        if not (eval_path / "eval_log_aggregated.json").exists():
            print("Evaluating...")
            with (LOGS_DIR / f"{save_name}_eval.txt").open("w") as outfile:
                cmd = f'accelerate launch --gpu-ids {gpu_id} mm/eval.py "model_path={model_dir.absolute()}"'
                subprocess.run(cmd, shell=True, stdout=outfile, stderr=outfile, env=current_env, check=True)
        print("Done")

    except KeyboardInterrupt:
        print("Experiment interrupted")
        exit()

    except Exception as e:
        print(f"Error running experiment with model {save_name}")
        print(e)


def run_finetuning(task: dict, gpu_id: int):
    save_name = "_".join(task.values()).replace("/", "_")

    print(f"{gpu_id}: Running task {save_name}")
    model_dir = Path(f"models/llava/{save_name}")

    try:
        if not (model_dir / "adapter_model.safetensors").exists():
            print("Finetuning...")
            train_task = task.copy()
            train_task["save_dir"] = model_dir.absolute()

            with (LOGS_DIR / f"{save_name}_ft.txt").open("w") as outfile:
                cmd = f"accelerate launch --gpu-ids {gpu_id} mm/finetune.py {task_args_to_str(train_task)}"
                print(cmd)
                subprocess.run(cmd, shell=True, stdout=outfile, stderr=outfile, env=current_env, check=True)

        eval_path = model_dir / "eval_results"
        if not (eval_path / "eval_log_aggregated.json").exists():
            print("Evaluating...")
            with (LOGS_DIR / f"{save_name}_eval.txt").open("w") as outfile:
                cmd = f'accelerate launch --gpu-ids {gpu_id} mm/eval.py "model_path={model_dir.absolute()}" "eval_task_ids=[3]"'
                subprocess.run(cmd, shell=True, stdout=outfile, stderr=outfile, env=current_env, check=True)
        print("Done")

    except KeyboardInterrupt:
        print("Experiment interrupted")
        exit()

    except Exception as e:
        print(f"Error running experiment with model {save_name}")
        print(e)


losses = [
    "LLMU",
    "scrub",
    "dpo",
    "RMU",
    "grad_diff_forget_ce_retain_ce",
    "retain_ft",
    "grad_ascent",
    "KL",
    "idk",
    "npo",
]


task_queue = queue.Queue()

for loss in losses:
    # for modality in ["text", "visual", "both"]:
    # if modality == "text":
    #     task_queue.put({"forget_loss": loss, "forget_split": "forget10", "forget_data_path": "locuslab/TOFU"})
    # elif modality == "visual":
    #     task_queue.put({"forget_loss": loss, "forget_split": "forget10", "forget_data_path": "therem/faces_v1"})
    # else:
    task_queue.put({"forget_loss": loss})


def worker(gpu_id):
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
            # run_finetuning(task, gpu_id)
            run_single_exp(task, gpu_id)
        except queue.Empty:
            break
        except Exception as e:
            print(f"Error running task on GPU {gpu_id}")
            print(e)
        finally:
            task_queue.task_done()


print(task_queue.qsize())

with ThreadPoolExecutor(max_workers=len(GPU_LIST)) as executor:
    futures = [executor.submit(worker, gpu_id) for gpu_id in GPU_LIST]
task_queue.join()
