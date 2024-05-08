import os
import json
from torch.utils.data import Dataset
from torchvision.datasets.utils import download_url
from PIL import Image
from data.utils import pre_caption
import random


captioning_instructions = [
    "A short image caption:",
    "A short image description:",
    "A photo of",
    "An image that shows",
    "Write a short description for the image.",
    "Write a description for the photo.",
    "Provide a description of what is presented in the photo.",
    "Briefly describe the content of the image.",
    "Can you briefly explain what you see in the image?",
    "Could you use a few words to describe what you perceive in the photo?",
    "Please provide a short depiction of the picture.",
    "Using language, provide a short account of the image.",
    "Use a few words to illustrate what is happening in the picture."
]


class TextCAPs(Dataset):
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.textcaps_root = 'data/textcaps'    # change to match data path

        if self.is_train:
            annotations = os.path.join(self.textcaps_root, 'TextCaps_0.1_train.json')
            self.annotation = json.load(open(annotations, 'r'))['data']
        else:
            annotations = os.path.join(self.textcaps_root, 'TextCaps_0.1_val.json')
            self.annotation = json.load(open(annotations, 'r'))['data']

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.textcaps_root, 'train_images', ann['image_id'] + '.jpg')
        image = Image.open(image_path).convert('RGB')
        # pick a random instruction
        random_id = random.randint(0, len(captioning_instructions) - 1)
        question = captioning_instructions[random_id]

        if self.is_train:
            weights = [1.0]
            # generate the ground truth caption
            caption = 'a picture of ' + pre_caption(ann['caption_str'], max_words=30)
            return image, question, question, [caption], weights, True
        else:
            return image, question, question, ann['image_id'], True


class COCO(Dataset):
    def __init__(self, is_train=True):
        ann_root = "data/coco"
        self.is_train = is_train
        if self.is_train:
            url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json'
            filename = 'coco_karpathy_train.json'
            download_url(url, ann_root)
        else:
            url = 'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json'
            filename = 'coco_karpathy_test.json'
            download_url(url, ann_root)
        self.annotation = json.load(open(os.path.join(ann_root, filename), 'r'))
        self.coco_root = "data/vqav2"
        self.max_words = 30

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.coco_root, ann['image'])
        image = Image.open(image_path).convert('RGB')
        # pick a random instruction
        random_id = random.randint(0, len(captioning_instructions) - 1)
        question = captioning_instructions[random_id]

        if self.is_train:
            weights = [1.0]
            # generate the ground truth caption
            caption = 'a picture of ' + pre_caption(ann['caption'], max_words=30)
            return image, question, question, [caption], weights, True
        else:
            img_id = ann['image'].split('/')[-1].strip('.jpg').split('_')[-1]
            return image, question, question, img_id, True

