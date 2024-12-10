import random

import datasets
import torch
from torch.utils.data import Dataset

from utils import add_dataset_index

IMAGE_CAPTION_QUESTIONS = [
    "What can you see in this picture?",
    "Tell me about the content of this image",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]


def convert_mm_data_to_model_format(processor, sample):
    # print("sample: ", sample)
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["question"]},
            ],
        },
    ]
    if sample.get("image", None) is not None:
        conversation[0]["content"].insert(0, {"type": "image"})

    formatted_question = processor.apply_chat_template(
        conversation, add_generation_prompt=True
    )
    # need number of question tokens for label masking
    num_question_tokens = len(
        processor.tokenizer.tokenize(formatted_question, add_special_tokens=True)
    )

    conversation.append(
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["answer"]},
            ],
        }
    )
    full_text = processor.apply_chat_template(conversation, add_generation_prompt=False)
    # print("full_text:", full_text)
    return full_text, num_question_tokens


class MMDatasetQA(Dataset):
    def __init__(
        self,
        data_path,
        processor,
        max_length=400,
        split=None,
        question_key="question",
        answer_key="answer",
        image_key="image",
    ):
        super(MMDatasetQA, self).__init__()
        self.processor = processor
        self.max_length = max_length
        self.data = datasets.load_dataset(data_path, split=split)
        self.data = add_dataset_index(self.data)
        self.qk = question_key
        self.ak = answer_key
        self.ik = image_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        itm = self.data[idx]
        res = {
            "question": itm[self.qk],
            "answer": itm[self.ak],
            "image": itm[self.ik],
        }
        return res


QUESTION_STRATEGIES = ("random_caption", "random_faces", "column")


class ImageCaptioningDataset(Dataset):
    def __init__(
        self,
        data_path,
        split=None,
        caption_key="caption",
        image_key="image",
        question_strategy="random_caption",
        question_key=None,
    ):
        super(ImageCaptioningDataset, self).__init__()
        self.data = datasets.load_dataset(data_path, split, split="train")
        self.data = add_dataset_index(self.data)
        self.ik = image_key
        self.ck = caption_key
        if question_strategy not in QUESTION_STRATEGIES:
            raise ValueError(
                f"Unknown question_strategy type: {question_strategy}, please choose from {QUESTION_STRATEGIES}"
            )

        if question_strategy == "column" and question_key is None:
            raise ValueError(
                "Question key must be provided when using question_strategy column"
            )
        self.question_strategy = question_strategy
        self.qk = question_key

    def __len__(self):
        return len(self.data)

    def __get_question(self, itm):
        if self.question_strategy == "random_caption":
            return random.choice(IMAGE_CAPTION_QUESTIONS)
        elif self.question_strategy == "random_faces":
            return "The name of the person on the image is "
        elif self.question_strategy == "column":
            return itm[self.qk]

    def __getitem__(self, idx):
        itm = self.data[idx]
        if isinstance(itm[self.ck], list):
            return [
                {
                    "idx": idx,
                    "image": itm[self.ik],
                    "answer": caption,
                    "question": self.__get_question(itm),
                }
                for caption in itm[self.ck]
            ]
        return {
            "idx": idx,
            "image": itm[self.ik],
            "answer": itm[self.ck],
            "question": self.__get_question(itm),
        }


class MMMixedDataset(Dataset):
    def __init__(
        self,
        data_path,
        split=None,
        caption_key="caption",
        image_key="image",
        question_key="question",
        answer_key="answer",
    ):
        super(MMMixedDataset, self).__init__()
        self.data = datasets.load_dataset(data_path, split)["train"]
        # self.data = add_dataset_index(self.data)
        self.ik = image_key
        self.ck = caption_key
        self.qk = question_key
        self.ak = answer_key

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        itm = self.data[idx]

        if itm[self.ik] is None:
            # this is a QA sample
            return {
                "image": None,
                "question": itm[self.qk],
                "answer": itm[self.ak],
            }
        else:
            return {
                "image": itm[self.ik],
                "question": random.choice(IMAGE_CAPTION_QUESTIONS),
                "answer": itm[self.ck],
            }


class MMMixedForgetDataset(Dataset):
    def __init__(
        self,
        forget_data_path,
        retain_data_path,
        forget_loss,
        retain_split,
        forget_split,
        caption_key="caption",
        image_key="image",
        question_key="question",
        answer_key="answer",
    ):
        super(MMMixedForgetDataset, self).__init__()
        self.forget_loss = forget_loss
        self.retain_split = retain_split
        self.forget_split = forget_split

        self.return_pairs = tuple()
        if self.forget_loss in ("dpo", "LLMU"):
            self.return_pairs = ("forget", "retain", "idk")
        elif self.forget_loss in ("idk",):
            self.return_pairs = ("idk", "retain")
        else:
            self.return_pairs = ("forget", "retain")

        if "idk" in self.return_pairs:
            self.idontknowfile = "data/idontknow.jsonl"
            self.idk_answers = open(self.idontknowfile, "r").readlines()

        self.forget_data = datasets.load_dataset(forget_data_path, self.forget_split)[
            "train"
        ]
        self.retain_data = datasets.load_dataset(retain_data_path, self.retain_split)[
            "train"
        ]
        # self.forget_data = add_dataset_index(self.forget_data)
        # self.retain_data = add_dataset_index(self.retain_data)
        self.image_key = image_key
        self.caption_key = caption_key
        self.question_key = question_key
        self.answer_key = answer_key

    def __len__(self):
        return len(self.forget_data)

    def _format_pair(self, pair):
        if pair.get(self.image_key, None) is None:
            # this is a QA sample
            return {
                "image": None,
                "question": pair[self.question_key],
                "answer": pair[self.answer_key],
            }
        else:
            return {
                "image": pair[self.image_key],
                "question": random.choice(IMAGE_CAPTION_QUESTIONS),
                "answer": pair[self.caption_key],
            }

    def __getitem__(self, idx):
        retain_idx = (idx + torch.randint(0, len(self.retain_data), (1,)).item()) % len(
            self.retain_data
        )
        forget_itm = self.forget_data[idx]
        retain_itm = self.retain_data[retain_idx]

        res = {"retain": self._format_pair(retain_itm)}

        if "forget" in self.return_pairs:
            res["forget"] = self._format_pair(forget_itm)

        if "idk" in self.return_pairs:
            forget_itm.update({self.answer_key: random.choice(self.idk_answers)})
            # for captions probably need different "I dont know" choices than for textual QA
            forget_itm.update({self.caption_key: random.choice(self.idk_answers)})
            res["idk"] = self._format_pair(forget_itm)

        return res


def mm_forget_data_collator_preprocessor(samples, processor, max_length):
    # samples is list of dicts with keys forget, retain, idk
    # preprocess each split (forget, retain, idk)
    # print("forget collator: samples", samples)
    res = {
        split: mm_data_collator_preprocessor(
            [s[split] for s in samples], processor, max_length
        )
        for split in samples[0].keys()
    }
    # print("forget collator: res", res)
    return res


def mm_data_collator_preprocessor(samples, processor, max_length, return_indices=False):
    #  samples is a list of lists, return also a dict with list of lists of tensors
    return_nested = False
    if isinstance(samples[0], list):
        return_nested = True
        list_size = len(samples[0])
        samples = sum(samples, [])

    images = [s["image"] for s in samples if s.get("image", None) is not None]

    full_texts, num_questions_tokens = [], []
    for sample in samples:
        full_test, num_question_tokens = convert_mm_data_to_model_format(
            processor, sample
        )
        full_texts.append(full_test)
        num_questions_tokens.append(num_question_tokens)

    inputs = processor(
        text=full_texts,
        images=images if images else None,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )

    # change label to -100 for question tokens
    labels = inputs.input_ids.clone()
    for num_question_tokens, label in zip(num_questions_tokens, labels):
        # find first non-pad token
        non_pad_tokens = (label != processor.tokenizer.pad_token_id).nonzero(
            as_tuple=True
        )[0]
        if len(non_pad_tokens) > 0:
            if processor.tokenizer.padding_side == "left":
                # Mask question tokens and left padding
                label[: non_pad_tokens[0] + num_question_tokens] = -100
            else:
                # Mask question tokens and right padding
                label[
                    non_pad_tokens[0] : non_pad_tokens[0] + num_question_tokens
                ] = -100
                label[non_pad_tokens[-1] + 1 :] = -100
        else:
            # If all tokens are padding, mask everything
            label[:] = -100

    inputs["labels"] = labels

    if return_indices:
        inputs["indices"] = torch.tensor([s["idx"] for s in samples])

    if return_nested:
        inputs = {
            key: torch.stack(
                [
                    inputs[key][i : i + list_size]
                    for i in range(0, len(inputs[key]), list_size)
                ]
            )
            for key in inputs.keys()
        }

    return inputs
