import os
import json
from PIL import Image
from torch.utils.data import Dataset
from data.utils import pre_question
from torchvision.datasets.utils import download_url
from collections import defaultdict


class VQA(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.vqa_root = 'data/vqav2'
        self.vg_root = 'data/vg'
        ann_root = 'data/vqav2'

        if self.is_train:
            urls = {'vqa_train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_train.json',
                    'vqa_val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_val.json',
                    'vg_qa': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/vg_qa.json'
                    }
            self.annotation = []
            for f in urls.keys():
                download_url(urls[f], ann_root)
                self.annotation += json.load(open(os.path.join(ann_root, '%s.json' % f), 'r'))
        else:
            download_url('https://storage.googleapis.com/sfr-vision-language-research/datasets/vqa_test.json', ann_root)
            self.annotation = json.load(open(os.path.join(ann_root, 'vqa_test.json'), 'r'))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]
        if ann['dataset'] == 'vqa':
            image_path = os.path.join(self.vqa_root, ann['image'])
        elif ann['dataset'] == 'vg':
            image_path = os.path.join(self.vg_root, ann['image'])
        else:
            raise NotImplementedError(f"dataset attribute of annotation should be vqa or vg and not {ann['dataset']}")

        image = Image.open(image_path).convert('RGB')
        question = pre_question(ann['question'])
        if not self.is_train:
            question_id = ann['question_id']
            return image, question, question, question_id, False

        else:
            if ann['dataset'] == 'vqa':
                answer_weight = defaultdict(lambda: 0)
                for answer in ann['answer']:
                    answer_weight[answer] += 1 / len(ann['answer'])

                answers = list(answer_weight.keys())
                weights = list(answer_weight.values())

            elif ann['dataset'] == 'vg':
                answers = [ann['answer']]
                weights = [0.2]
            else:
                raise NotImplementedError(f"dataset attribute of annotation should be vqa or vg and not {ann['dataset']}")

            return image, question, question, answers, weights, False


class VizWiz(Dataset):
    def __init__(self, **kwargs):
        self.vqa_root = 'data/vizwiz'
        ann_root = 'data/vizwiz'
        self.annotation = json.load(open(os.path.join(ann_root, 'test.json'), 'r'))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.vqa_root, 'test', ann['image'])
        image = Image.open(image_path).convert('RGB')
        instruct = pre_question(ann['question'])
        question = instruct
        question_id = ann['image']
        return image, question, instruct, question_id, False
