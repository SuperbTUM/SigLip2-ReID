import os
import copy
import random
import torch
import numpy as np
from collections import defaultdict
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from base_dataset import ImageDataset
from constants import *
from market import Market1501
from veri import VeRi


class LocalizedGray(object):
    def __init__(self, p=0.3, scale=(0.1, 0.4)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if torch.rand(1) > self.p:
            return img

        C, H, W = img.shape
        area = H * W
        target_area = torch.empty(1).uniform_(*self.scale).item() * area

        h = int(torch.sqrt(torch.tensor(target_area)))
        w = h
        y = torch.randint(0, H - h + 1, (1,))
        x = torch.randint(0, W - w + 1, (1,))

        gray = img[:, y:y+h, x:x+w].mean(dim=0, keepdim=True)
        img[:, y:y+h, x:x+w] = gray.repeat(3, 1, 1)
        return img

class PKsamplerWithLabels:
    """
    A P-K sampler that yields a tuple of (indices, labels) for each batch.
    
    NOTE: This sampler is intended for manual iteration, not for use with
    the DataLoader's `batch_sampler` argument.
    
    Args:
        labels (list): A list containing the label for each sample in the dataset.
        p (int): The number of classes per batch.
        k (int): The number of samples per class.
    """
    def __init__(self, labels, p, k):
        self.p = p
        self.k = k
        self.batch_size = self.p * self.k
        
        # Create a dictionary mapping each label to its indices
        self.labels_to_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            self.labels_to_indices[label].append(idx)
        
        self.unique_labels = list(self.labels_to_indices.keys())
        
        # Filter out classes with fewer than K samples
        self.valid_labels = [
            label for label in self.unique_labels 
            if len(self.labels_to_indices[label]) >= self.k
        ]

    def __iter__(self):
        # Shuffle the list of valid class labels for randomness between epochs
        random.shuffle(self.valid_labels)
        
        batch_indices = []
        batch_labels = [] 
        
        for label in self.valid_labels:
            indices = self.labels_to_indices[label]
            sampled_indices = random.sample(indices, self.k)
            
            # Add the indices and their corresponding labels to the batch lists
            batch_indices.extend(sampled_indices)
            batch_labels.extend([label] * self.k) 
            
            # If the batch is full, yield the data and reset
            if len(batch_indices) == self.batch_size:
                yield batch_indices, batch_labels
                batch_indices = []
                batch_labels = []

    def __len__(self):
        return len(self.valid_labels) // self.p

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list) #dict with list value
        #{783: [0, 5, 116, 876, 1554, 2041],...,}
        for index, (_, pid, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length

def create_dataloader(dataset_name, input_size, type, augmented, use_ai_prompts=False, dual_branch=False):
    weak_augment_preprocessing = T.Compose([
            T.Resize(input_size),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])
    strong_augment_preprocessing = T.Compose([
            T.Resize(input_size),
            T.RandomHorizontalFlip(0.5),
            # T.ColorJitter(
            #     brightness=0.2,
            #     contrast=0.2,
            #     saturation=0.15,
            #     hue=0.1,
            # ),
            T.Pad(10),
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
            T.RandomErasing(p=0.25),
            LocalizedGray(p=0.25, scale=(0.1, 0.3)),
        ])
    if augmented:
        preprocessing = T.Compose([
            T.Resize(input_size),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop(input_size),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5),
            T.RandomErasing(p=0.25),
            LocalizedGray(p=0.25, scale=(0.1, 0.3)),
        ])
    else:
        preprocessing = T.Compose([
            T.Resize(input_size),
            T.ToTensor(),
            T.Normalize(mean=0.5, std=0.5)
        ])
    if dataset_name == "Market1501":
        dataset = Market1501(verbose=False)
    elif dataset_name == "veri":
        dataset = VeRi(verbose=False)
    if type == "train":
        if dual_branch:
            preprocessed_dataset = ImageDataset(dataset.train, weak_augment_preprocessing, strong_augment_preprocessing)
        else:
            preprocessed_dataset = ImageDataset(dataset.train, preprocessing)
        if use_ai_prompts:
            ai_prompts = []
            if os.path.exists(f"prompts_{dataset_name}_full.txt"):
                with open(f"prompts_{dataset_name}_full.txt", "r", encoding="utf-8") as f:
                    ai_prompts += [prompt.strip() for prompt in f.readlines()]
        else:
            ai_prompts = None
    else:
        preprocessed_dataset = ImageDataset(dataset.query + dataset.gallery, preprocessing)
    dataloader = DataLoader(preprocessed_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=not augmented, 
                            sampler=RandomIdentitySampler(dataset.train, BATCH_SIZE, N_INSTANCE) if augmented else None,
                            num_workers=N_WORKER)
    return dataloader, None if type == "train" else len(dataset.query), dataset.num_train_pids if type == "train" else None, ai_prompts if type == "train" else None
