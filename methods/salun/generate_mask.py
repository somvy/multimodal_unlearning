
import sys
import os
import hydra
import gc 

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm.auto import tqdm 




def save_gradient_ratio(forget_loader, model, criterion, cfg):
    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.unlearn_lr,
        # momentum=args.momentum,
        weight_decay=cfg.weight_decay,
    )

    gradients = {}
    model.eval()

    for name, param in model.named_parameters():
        gradients[name] = 0

    for i, batch in enumerate(tqdm(forget_loader)):
        input_ids, labels, attention_mask, indices = batch

        # compute output
        device=model.device

        loss = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device), labels=labels.to(device)).loss
        optimizer.zero_grad()
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data


    with torch.no_grad():
        for name in gradients:
            gradients[name] = gradients[name].abs_().cpu().to(torch.float16)

    print("Saving gradients...")
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    def process_threshold(threshold, gradients):
        # sorted_dict_positions = {}
        hard_dict = {}

        # Initialize the position counter
        start_index = 0

        for key, tensor in gradients.items():
            num_elements = tensor.numel()

            # Flatten and sort tensor
            flat_tensor = tensor.flatten()
            sorted_tensor, indices = torch.sort(flat_tensor, descending=True)

            # Find the threshold index
            threshold_index = int(threshold * num_elements)

            # Create hard threshold tensor
            threshold_tensor = torch.zeros_like(flat_tensor, dtype=torch.int8)
            threshold_tensor[indices[:threshold_index]] = 1

            # Reshape tensors to original shape
            # sorted_positions = indices.argsort().reshape(tensor.shape)
            # sorted_dict_positions[key] = sorted_positions
            hard_dict[key] = threshold_tensor.reshape(tensor.shape)

            del flat_tensor, sorted_tensor, indices, threshold_tensor
            gc.collect()

            start_index += num_elements

        return hard_dict
    
    for i in tqdm(threshold_list):
        hard_dict = process_threshold(i, gradients)
        torch.save(hard_dict, os.path.join(cfg.save_dir, f"with_{i}.pt"))
    

@hydra.main(version_base=None, config_path=".", config_name="generate_mask")
def main(cfg):
    os.makedirs(cfg.save_dir, exist_ok=True)

    seed = cfg.seed
    device = "cuda"
    set_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path,
                                                torch_dtype=torch.bfloat16
                                                 ).to(device)
    model.eval()


    sys.path.insert(0, hydra.utils.get_original_cwd())

    from data_module import TextDatasetQA
    forget_dataset = TextDatasetQA(cfg.forget_dataset, tokenizer=tokenizer, model_family=cfg.model_family, 
                                   max_length=cfg.max_length, 
                                   split=cfg.forget_split)

    forget_loader = torch.utils.data.DataLoader(
        forget_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"number of forget dataset {len(forget_dataset)}")

    criterion = nn.CrossEntropyLoss()

    save_gradient_ratio(forget_loader, model, criterion, cfg)


if __name__ == "__main__":
    main()
