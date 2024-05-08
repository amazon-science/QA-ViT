import torch
from data.multitask_dataset import MultiTaskDataset
from data.documents import DocVQA, InfoVQA
from data.vqa import VQA, VizWiz
from data.captioning import COCO, TextCAPs
from data.stvqa import TextVQA, STVQA
from collections import defaultdict


def vt5_collate_train_fn(batch, tokenizer, processor, **kwargs):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    instruct_list = []
    for image, question, instruct, answer, weights, _ in batch:
        if image.size[0] == 1 or image.size[1] == 1:
            continue    # skip illegal image
        image_list.append(image)
        question_list.append(question)
        instruct_list.append(instruct)
        answer_list.extend(answer)
        weight_list += weights
        n.append(len(answer))
    # tokenize
    questions = tokenizer(question_list, return_tensors="pt", padding='longest', truncation=True, max_length=30)
    answers = tokenizer(answer_list, return_tensors="pt", padding='longest', truncation=True,  max_length=30)
    # ensure no penalty for pad tokens and for the captioning prefix
    answers = answers.input_ids.masked_fill(answers.input_ids == tokenizer.pad_token_id, -100)
    cap_prompt_len = len(tokenizer('a picture of ', add_special_tokens=False).input_ids)
    is_cap = ['a picture of ' in ans for ans in answer_list]
    answers[is_cap, :cap_prompt_len] = -100
    return {
        'images': processor(images=image_list, return_tensors="pt"),
        'n': torch.Tensor(n),   # indicates how many questions-answers correspond to each image
        'weights': torch.Tensor(weight_list),
        'input_ids': questions.input_ids,
        'attention_mask': questions.attention_mask,
        'labels': answers,
        'instructions_list': instruct_list
    }


def vt5_collate_test_fn(batch, tokenizer, processor, **kwargs):
    image_list, question_list, question_id_list = [], [], []
    instruct_list = []
    for image, question, instruct, question_id, _ in batch:
        image_list.append(image)
        question_list.append(question)
        instruct_list.append(instruct)
        question_id_list.append(question_id)
    # tokenize
    questions = tokenizer(question_list, return_tensors="pt", padding='longest', truncation=True, max_length=30)
    return {
        'images': processor(images=image_list, return_tensors="pt"),
        'input_ids': questions.input_ids,
        'attention_mask': questions.attention_mask,
    }, question_id_list, instruct_list


def get_datasets(config, eval=False):
    text2dataset = defaultdict()
    text2dataset['multitask'] = MultiTaskDataset
    text2dataset['textvqa'] = TextVQA
    text2dataset['vqa'] = VQA
    text2dataset['coco'] = COCO
    text2dataset['textcaps'] = TextCAPs
    text2dataset['stvqa'] = STVQA
    text2dataset['docvqa'] = DocVQA
    text2dataset['infovqa'] = InfoVQA
    text2dataset['vizwiz'] = VizWiz

    # initialize the training dataset
    train_set = None
    if not eval:
        train_set = text2dataset[config['dataset_train']]()
    # initialize the evaluation dataset
    dataset_val = text2dataset[config['dataset_eval']]
    val_set = dataset_val(is_train=False)

    return train_set, val_set



