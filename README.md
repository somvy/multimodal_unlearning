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


## Citing

If you find our dataset useful, please cite:

```
@inproceedings{clear,
    title = "{CLEAR}: Character Unlearning in Textual and Visual Modalities",
    author = "Dontsov, Alexey  and
      Korzh, Dmitrii  and
      Zhavoronkin, Alexey  and
      Mikheev, Boris  and
      Bobkov, Denis  and
      Alanov, Aibek  and
      Rogov, Oleg  and
      Oseledets, Ivan  and
      Tutubalina, Elena",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.1058/",
    doi = "10.18653/v1/2025.findings-acl.1058",
    pages = "20582--20603",
    ISBN = "979-8-89176-256-5",
    abstract = "Machine Unlearning (MU) is critical for removing private or hazardous information from deep learning models. While MU has advanced significantly in unimodal (text or vision) settings, multimodal unlearning (MMU) remains underexplored due to the lack of open benchmarks for evaluating cross-modal data removal. To address this gap, we introduce CLEAR, the first open-source benchmark designed specifically for MMU. CLEAR contains 200 fictitious individuals and 3,700 images linked with corresponding question-answer pairs, enabling a thorough evaluation across modalities. We conduct a comprehensive analysis of 11 MU methods (e.g., SCRUB, gradient ascent, DPO) across four evaluation sets, demonstrating that jointly unlearning both modalities outperforms single-modality approaches. The dataset is available at [link](https://huggingface.co/datasets/therem/CLEAR)"
}
```

