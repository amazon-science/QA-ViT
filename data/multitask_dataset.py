from torch.utils.data import Dataset
import math
from data.stvqa import TextVQA, STVQA, OCRVQA
from data.vqa import VQA
from data.captioning import COCO, TextCAPs
from data.documents import DocVQA, InfoVQA, ChartQA
import numpy as np


class MultiTaskDataset(Dataset):
    def __init__(self, **kwargs):
        self.datasets_list = [
            VQA(),
            STVQA(),
            TextVQA(),
            OCRVQA(),
            TextCAPs(),
            COCO(),
            DocVQA(),
            InfoVQA(),
            ChartQA()
        ]

        # Weight according to square root of the sizes
        dataset_proba = []
        for dataset in self.datasets_list:
            dataset_proba.append(math.floor(math.sqrt(len(dataset))))
        self.dataset_proba = [proba / sum(dataset_proba) for proba in dataset_proba]
        assert abs(sum(self.dataset_proba) - 1.0) < 0.001

    def __len__(self):
        data_len = 0
        for dataset in self.datasets_list:
            data_len += len(dataset)
        return data_len

    def __getitem__(self, index):
        random_dataset_id = np.random.choice(len(self.dataset_proba), p=self.dataset_proba)
        data_sample_id = index % len(self.datasets_list[random_dataset_id])
        return self.datasets_list[random_dataset_id][data_sample_id]