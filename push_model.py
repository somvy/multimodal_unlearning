from transformers import AutoModelForPreTraining, AutoTokenizer

method = "RMU"
model_name = f"models/llava/ft_full+tofu_epoch3_lr1e-05__wd0.01_lora/{method}_forget10+tofu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForPreTraining.from_pretrained(model_name)

push_name = f"therem/llava-1.5-7b-CLEAR-forget-{method}"
model.push_to_hub(push_name)
tokenizer.push_to_hub(push_name)
