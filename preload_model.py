from transformers import AutoModelForPreTraining, AutoTokenizer

# model_name = "locuslab/tofu_ft_llama2-7b"
# model_name = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "NousResearch/Llama-2-7b-chat-hf"
model_name = "models/llava/ft_full+tofu_epoch3_lr1e-05__wd0.01_lora"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForPreTraining.from_pretrained(model_name)

model.push_to_hub("therem/ft_fulltext_epoch3_lr1e-05__wd1e-2_lora")
tokenizer.push_to_hub("therem/ft_fulltext_epoch3_lr1e-05__wd1e-2_lora")
# save_dir = "models/llama2-7b-chat"
# tokenizer.save_pretrained(save_dir)
# model.save_pretrained(save_dir)