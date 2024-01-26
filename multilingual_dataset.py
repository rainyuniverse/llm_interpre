from typing import Iterator
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler
import random
import torch
import math

class MultilingualDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.lang_pairs = list(data_dict.keys())

        self.dataset_lengths = len(self.data_dict[self.lang_pairs[0]])
        self.lang_pair_num = len(self.lang_pairs)

    def __len__(self):
        # 假设所有语言对都有相同数量的样本
        return len(self.data_dict[self.lang_pairs[0]])
    
    def __getitem__(self, index):
        lang_pair = index[0]
        sample_index = index[1]
        sample = self.data_dict[lang_pair][sample_index]

        # input_ids = sample["input_ids"] if "input_ids" in sample.keys() else None
        # attention_mask = sample["attention_mask"] if "attention_mask" in sample.keys() else None
        # labels = sample["labels"] if "labels" in sample.keys() else None

        # return {"lang_pair": lang_pair, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        return {"lang_pair": lang_pair, "data": sample}

class MultilingualBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        batch = list(range(len(self.dataset)))
        # 每个语言对的数据量（假设所有语言对的数据量相等，所以仅计算第一个语言对的数据量）
        per_lang_pair_length = 0
        for i in batch:
            if self.dataset[i]["lang_pair"] != self.dataset[i + 1]["lang_pair"]:
                per_lang_pair_length = i + 1
                break
        
        # 语言对数量
        lang_pairs_num = len(batch) // per_lang_pair_length
        # 初始化二维列表，二维列表中的每个子列表存放一个语言对的数据索引
        lang_pairs_indices = []
        for i in range(lang_pairs_num):
            lang_pairs_indices.append(batch[i * per_lang_pair_length:(i + 1) * per_lang_pair_length])

        # 打乱每个子列表中的索引顺序
        lang_pairs_indices = [random.sample(indices, len(indices)) for indices in lang_pairs_indices]

        for start in range(0, len(lang_pairs_indices[0]), self.batch_size):
            for j in range(lang_pairs_num):
                yield lang_pairs_indices[j][start: start + self.batch_size]     


    def __len__(self) -> int:
        batch = list(range(len(self.dataset)))
        # 每个语言对的数据量（假设所有语言对的数据量相等，所以仅计算第一个语言对的数据量）
        per_lang_pair_length = 0
        for i in batch:
            if self.dataset[i]["lang_pair"] != self.dataset[i + 1]["lang_pair"]:
                per_lang_pair_length = i + 1
                break
        
        # 语言对数量
        lang_pairs_num = len(batch) // per_lang_pair_length
        # 初始化二维列表，二维列表中的每个子列表存放一个语言对的数据索引
        lang_pairs_indices = []
        for i in range(lang_pairs_num):
            lang_pairs_indices.append(batch[i * per_lang_pair_length:(i + 1) * per_lang_pair_length])

        return (math.ceil(per_lang_pair_length / self.batch_size)) * lang_pairs_num


def collate_fn(batch):
    lang_pair = [sample["lang_pair"] for sample in batch][0]
    input_ids = [sample["input_ids"] for sample in batch]
    attention_mask = [sample["attention_mask"] for sample in batch]
    labels = [sample["labels"] for sample in batch]
    return {"lang_pair": lang_pair, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


if __name__ == "__main__":
    data_dict = {
        'lang1': ['data1', 'data2', 'data3', 'data4', 'data5', 'data6'],
        'lang2': ['data7', 'data8', 'data9', 'data10', 'data11', 'data12'],
    }
    multilingual_dataset = MultilingualDataset(data_dict)
    batch_size = 4
    batch_sampler = MultilingualBatchSampler(multilingual_dataset, batch_size)
    data_loader = DataLoader(multilingual_dataset, batch_sampler=batch_sampler)

    for batch in data_loader:
        print(batch)

