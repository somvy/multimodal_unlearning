This repo contains experiments for the CLEAR  benchmark - Character Unlearning in Textual and Visual Modalities.
It mearsures the effectiveness of unlearning methods in mutimodal setup. 

For details check the [**Arxiv Paper**](https://arxiv.org/abs/2410.18057)

Below are instructions for reproducing the experiments on multimodal model (LLAVA) - first finetuning the model on our dataset, then forgeting using different unlearning methods.

For LLM unlearning run the same scripts from `nlp` folder. They are structured similarly as multimodal experiments.


For CV only experiments, check out `cv` folder and `cv/README.md`

## Create environment
(tested with python 3.10 ) 

Install torch with cuda (may be specific to your platform)

        pip3 install torch

Ensure [nvidia-toolkit](https://developer.nvidia.com/cuda-toolkit) is installed by 
        
        nvcc --version

If not, install it 

Then install transformers, accelerate, etc

Manually:

        pip install "transformers[torch]" datasets accelerate peft deepspeed scipy 


        pip install hydra-core omegaconf setuptools wandb natsort pillow 

Or from req.txt file:

        pip install -r req.txt


For flash attention: 
        
        pip install flash-attn --no-build-isolation

For adam_8bit (consume significantly less gpu memory) you need to install [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

        pip install bitsandbytes

## Finetune on CLEAR
* edit config in config/mm/finetune.yaml, specify model, split, etc

* set 

        export PYTHONPATH=.

* then run 
      
        CUDA_VISIBLE_DEVICES=1 accelerate launch  mm/finetune.py

## Forget 

* edit config in config/mm/forget.yaml
* run 

        CUDA_VISIBLE_DEVICES=1 accelerate launch mm/forget.py


## Evaluate 

* edit config in config/mm/eval.yaml

Specify `model_path` with desired checkpoint to run
* run 

        CUDA_VISIBLE_DEVICES=1 accelerate launch mm/eval.py


## Calculate metrics

**Important** - to calculate "Forget Quality" metric we need a "gold" retain model, 
so we need to finetune and evaluate it. To do this, edit the config/mm/finetune.yaml again, change only the `split` field from "full" to "retainN"
For example, if you are forgetting 10 percent, the "gold" model should be trained on retain90.  

Finetune the "gold" model, then evaluate it. Remember, where the evaluation files were saved (usually `eval_reults` subfolder of model's folder).


* Edit config/aggregate_eval_stat.yaml

Change `retain_result` to `eval_log_aggregated.json` of gold retain model (this file is a result of evaluate step)

Change `ckpt_result` to `eval_log_aggregated.json` of model, which you are evaluating. 
Change `method_name`, `submitted_by` and `save_file`
Run - 

    python aggregate_eval_stat.py

The result of the experiment will be available in save_file


