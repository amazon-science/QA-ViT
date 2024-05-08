import os
import json
from PIL import Image
from torch.utils.data import Dataset
from data.utils import pre_question
from collections import defaultdict
import random


class DocVQA(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.data_root = os.path.join('data/DocVQA', "train" if is_train else "test")

        if self.is_train:
            train_annotations = 'data/DocVQA/train/train_v1.0.json'
            self.annotation = json.load(open(train_annotations, 'r'))['data']
        else:
            test_annotations = 'data/DocVQA/test/test_v1.0.json'
            self.annotation = json.load(open(test_annotations, 'r'))['data']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.data_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        question = pre_question(ann['question'])
        if not self.is_train:
            question_id = ann['questionId']
            return image, question, question, question_id, False

        answer_weight = defaultdict(lambda: 0)
        for answer in ann['answers']:
            answer_weight[answer] += 1 / len(ann['answers'])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return image, question, question, answers, weights, False


class InfoVQA(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.data_root = os.path.join('data/infovqa')
        self.annotation = json.load(open('data/infovqa/infographicsVQA_train_v1.0.json', 'r'))['data']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.data_root , ann['image_local_name'])
        image = Image.open(image_path).convert('RGB')
        if not self.is_train:
            question = pre_question(ann['question'])
            question_id = ann['question_id']
            return image, question, question, question_id, False
        else:
            random_question = random.randint(0, len(ann['question']) - 1)
            question = pre_question(ann['question'])
            answer_weight = defaultdict(lambda: 0)
            for answer in ann['answers'][random_question]:
                answer_weight[answer] += 1 / len(ann['answers'][random_question])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            return image, question, question, answers, weights, False


class ChartQA(Dataset):
    def __init__(self, **kwargs):
        self.data_root = os.path.join('data/ChartQA', "train")

        train_annotations = 'data/ChartQA/train/train_human.json'
        self.annotation = json.load(open(train_annotations, 'r'))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.data_root, 'png', ann['imgname'])
        image = Image.open(image_path).convert('RGB')
        question = pre_question(ann['query'])
        answer_weight = defaultdict(lambda: 0)
        for answer in ann['label']:
            answer_weight[answer] += 1 / len(ann['label'])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        return image, question, question, answers, weights, False
