import os
import json
from PIL import Image
from torch.utils.data import Dataset
from data.utils import pre_question
from collections import defaultdict
import random

vqa_templates = [
    "{}",
    "Question: {}",
    "{} A short answer to the question is",
    "Q: {} A:",
    "Question: {} Short answer:",
    "Given the image, answer the following question with no more than three words. {}",
    'Based on the image, respond to this question with a short answer: {}. Answer:',
    "Use the provided image to answer the question: {} Provide your answer as short as possible:",
    'What is the answer to the following question? "{}"',
    'The question "{}" can be answered using the image. A short answer is',
]


class TextVQA(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.textvqa_root = 'data/textvqa/train_images'

        if self.is_train:
            train_annotations = 'data/textvqa/train/TextVQA_0.5.1_train.json'
            self.annotation = json.load(open(train_annotations, 'r'))['data']
        else:
            train_annotations = 'data/textvqa/train/TextVQA_0.5.1_val.json'
            self.annotation = json.load(open(train_annotations, 'r'))['data']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.textvqa_root, (ann['image_id'] + '.jpg'))

        image = Image.open(image_path).convert('RGB')
        question = pre_question(ann['question'])
        if not self.is_train:
            question_id = ann['question_id']
            return image, question, question, question_id, False
        else:
            answer_weight = defaultdict(lambda: 0)
            for answer in ann['answers']:
                answer_weight[answer] += 1 / len(ann['answers'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            return image, question, question, answers, weights, False


class STVQA(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.stvqa_root = 'data/stvqa'

        if self.is_train:
            train_annotations = 'data/stvqa/train_task_3.json'
            self.annotation = json.load(open(train_annotations, 'r'))['data']
        else:
            test_annotations = 'data/stvqa/test/test_task_3.json'
            self.annotation = json.load(open(test_annotations, 'r'))['data']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        # image_path = os.path.join(self.stvqa_root, "train" if self.is_train else "test", (ann['file_path']))
        image_path = os.path.join(self.stvqa_root, (ann['file_path']))
        image = Image.open(image_path).convert('RGB')
        while image.size[0] == 1 or image.size[1] == 1:
            img_id = ann['image_id']
            print(f'encountered invalid image - img_id {img_id}')
            index = random.randint(0, len(self.annotation) - 1)
            ann = self.annotation[index]
            image_path = os.path.join(self.stvqa_root, (ann['file_path']))
            image = Image.open(image_path).convert('RGB')
        question = pre_question(ann['question'])
        if not self.is_train:
            question_id = ann['question_id']
            return image, question, question, question_id, False
        else:
            answer_weight = defaultdict(lambda: 0)
            for answer in ann['answers']:
                answer_weight[answer] += 1 / len(ann['answers'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())

            return image, question, question, answers, weights, False

#
# class OCRVQA(Dataset):
#     def __init__(self, **kwargs):
#         self.split = "train"
#         self.ocrvqa_root = 'data/ocrvqa'
#         annotations = json.load(open(os.path.join(self.ocrvqa_root, 'dataset.json')))
#         self.annotation = []
#         for img_id in annotations.keys():
#             if annotations[img_id]['split'] == 1:   # use only the train split
#                 datum = {
#                     'image_id': img_id,
#                     'questions': annotations[img_id]['questions'],
#                     'answers': annotations[img_id]['answers'],
#                 }
#                 self.annotation.append(datum)
#
#     @staticmethod
#     def get_confidence_threshold(word):
#         return 10 if len(word) < 8 else 3
#
#     def get_image_path(self, img_id):
#         suffix = [
#             '.jpg',
#             '.png',
#             '.gif'
#         ]
#         for suff in suffix:
#             p = os.path.join(self.ocrvqa_root, 'images', img_id + suff)
#             if os.path.exists(p):
#                 return p
#         raise IndexError(f"{img_id} not found")
#
#
#
#     def __len__(self):
#         return len(self.annotation)
#
#     def __getitem__(self, index):
#         ann = self.annotation[index]
#         image_path = self.get_image_path(ann['image_id'])
#         image = Image.open(image_path).convert('RGB')
#         while image.size[0] == 1 or image.size[1] == 1:
#             img_id = ann['image_id']
#             print(f'encountered invalid image - img_id {img_id}')
#             index = random.randint(0, len(self.annotation) - 1)
#             ann = self.annotation[index]
#             image_path = self.get_image_path(ann['image_id'])
#             image = Image.open(image_path).convert('RGB')
#
#         random_id = random.randint(0, len(ann['questions']) - 1)     # pick random question and answer from list
#         question = ann['questions'][random_id]
#         random_prefix = random.randint(0, len(vqa_templates) - 1)   # not used eventually
#         question = pre_question(question)
#
#         answers = [ann['answers'][random_id]]
#         weights = [1.0]
#         if self.split == 'train':
#             return image, question, question, answers, weights, False
#         else:
#             return image, question, question, str(ann['image_id']) + '_' + question, False

class OCRVQA(Dataset):
    def __init__(self, only_answers=False, **kwargs):
        self.only_answers = only_answers
        self.data_root = '/home/ganz/data/MM/ocrvqa/images'
        ann_path = '/home/ganz/data/MM/ocrvqa/dataset.json'
        annotation = json.load(open(os.path.join(ann_path), 'r'))
        self.annotation = []
        for ann in annotation.values():
            if ann['split'] == 1:
                # idx = random.randint(0, len(ann['questions']) - 1)
                elem = {
                    'image': ann['imageURL'].split('/')[-1],
                    'questions': ann['questions'],
                    'answers': ann['answers']
                }
                self.annotation.append(elem)
                # for q, a in zip(ann['questions'], ann['answers']):
                #     elem = {
                #         'image': ann['imageURL'].split('/')[-1],
                #         'question': q,
                #         'answer': a
                #     }
                #     self.annotation.append(elem)
        a = 1

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.data_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        random_id = random.randint(0, len(ann['questions']) - 1)  # pick random question and answer from list
        question = ann['questions'][random_id]
        random_prefix = random.randint(0, len(vqa_templates) - 1)  # not used eventually
        question = pre_question(question)

        answers = [ann['answers'][random_id]]
        weights = [1.0]

        # return image, question, answer
        return image, question, question, answers, weights, False
