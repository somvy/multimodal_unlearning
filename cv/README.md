0) Create a Conda environment and install the required libraries:
```
conda create -n CLEAR python=3.7
conda activate CLEAR
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

1) Prepare Train-Test Split for Pretraining on CelebA
```
python prepare_celeb_dataset.py
```

2) Pretrain the Model on CelebA
```
python pretrain.py
python pretrain.py
```

4) Evaluate the Pretrained Model on the CelebA dataset
```
python score.py --num_enroll 5 --method pretrained  --dataset celebrity
```

3) Prepare Splits for CLEAR Fine-tuning
```
python prepare_vtofu_dataset.py
```

5) Fine-tune the Model on CLEAR
```
python run_finetune.py --start_split_idx 0 --stop_split_idx 128
```

6) Perform the Unlearning Procedure
```
python run_unlearn.py --method retrain --start_split_idx 0 --stop_split_idx 128 --forget_size 10
```

7) Attack the Unlearned Model Using U-LIRA
```
python attack.py --method retrain --net resnet18 --forget_size 10 --attack ulira --plot_distr
```

### Methods and their notation

| script       | method        |
|:-------------|:--------------|
| finetuned.py | Orginal       |
| retrain.py   | Gold          |
| scrub.py     | SCRUB$_{bio}$ |
| twins.py     | Twins         |
| dpo.py       | DPO           |
| finetune.py  | Retain FT     |
| llmu.py      | LLMU          |
| rmu.py       | RMU           |
| sparsity.py  | Sparsity      |