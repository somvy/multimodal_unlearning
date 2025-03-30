import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES_WITH_TEACHER = ("DPO", "SCRUB", "RMU", "LLMU", "SKU", "NPO")


def loss_needs_teacher(loss_type: str):
    return "KL" in loss_type or loss_type.upper() in LOSSES_WITH_TEACHER


def has_lora_adapter(model):
    return any(True for name, param in model.named_parameters() if "lora" in name)


def logits2probs(logits, log=True):
    if log:
        probs = F.log_softmax(logits, dim=-1)
    else:
        probs = F.softmax(logits, dim=-1)
    return probs.view(-1, logits.shape[-1])


def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)

    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()

    return cache[0]


def remove_image_tokens(input_ids, logits, image_token_id):
    # the output logits from the model come in extended version with the image pathch tokens included (aroung 500)
    search = (input_ids == image_token_id).nonzero(as_tuple=True)
    has_image_tokens = search[0].tolist()
    image_tokens_indices = search[1].tolist()
    batch_size, seq_len = input_ids.shape
    new_logits = []

    for i in range(batch_size):
        if i not in has_image_tokens:
            # there are no image tokens in this sample, remove paddings
            # assume thath the text tokens are at the end of the sequence (left padding as recommended)
            sample = logits[i, -seq_len:, :]
            new_logits.append(sample)
            continue

        start_image_tokens = image_tokens_indices[has_image_tokens.index(i)]
        left_num_tokens = start_image_tokens
        right_num_tokens = seq_len - start_image_tokens
        sample = torch.cat([logits[i, :left_num_tokens, :], logits[i, -right_num_tokens:, :]])
        new_logits.append(sample)
    res = torch.stack(new_logits)
    return res


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1, -2), shifted_labels).sum(dim=-1)

    return loss


# def get_mm_batch_loss(output, labels, processor):
#  image_token_idxs = (batch["input_ids"] == model.config.image_token_index).nonzero(as_tuple=True)[1]
