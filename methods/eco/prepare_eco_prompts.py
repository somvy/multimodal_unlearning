from eco.attk_model import AttackedModel
from eco.model import HFModel
from eco.classifier import PromptClassifier, TokenClassifier
from transformers import GenerationConfig
from datasets import load_dataset
from eco.main import get_eco_model

# corrputp forget prompts and generate new ones on corrupted
setup = {
    "batch_size": 32,
    "classifier_threshold": 0.99,
    "model_name": "llama2-7b",
    "corrupt_method": "zero_out_top_k",
    "corrupt_args": {"dims": 1000},
}
model_path = "models/llama2-7b-tofu"


model = get_eco_model(setup, model_path)

config = GenerationConfig(
    max_length=200,
    # top_p=0.9,
    temperature=0.7,
    # do_sample=True,
    num_return_sequences=1,
)


def generate_eco_answers(itms):
    prompts = itms["question"]
    inputs = model.tokenizer(
        prompts, return_tensors="pt", padding="longest", truncation=True
    )
    inputs = {
        "input_ids": inputs["input_ids"].to(model.device),
        "attention_mask": inputs["attention_mask"].to(model.device),
    }
    outputs = model.generate(**inputs, prompts=prompts, generation_config=config)
    answers = model.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    itms["orig_answer"] = itms["answer"]
    itms["answer"] = answers
    return itms


dataset = load_dataset("locuslab/TOFU", "forget10")["train"]
new_dataset = dataset.map(generate_eco_answers, batched=True, batch_size=64)
print(new_dataset[0])
new_dataset.save_to_disk("data/eco_prompts_forget10")
