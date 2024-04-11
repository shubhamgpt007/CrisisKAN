import os
import torch
import numpy as np
from imageio import imread
from PIL import Image
import glob

from termcolor import colored, cprint

from preprocess import clean_text

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision.utils as vutils
from torchvision import datasets

from transformers import BertTokenizer, XLNetTokenizer, ElectraTokenizer, AlbertTokenizer
from preprocess import clean_text

from base_dataset import BaseDataset
from base_dataset import expand2square

from paths import dataroot

task_dict = {
    'task1': 'informative',
    'task2_full': 'humanitarian',
    'task2': 'humanitarian',
    'task3' : 'damage'
}

labels_task1 = {
    'informative': 1,
    'not_informative': 0
}

labels_task2_full = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 6,
    'missing_or_found_people': 7,
}

labels_task2 = {
    'infrastructure_and_utility_damage': 0,
    'not_humanitarian': 1,
    'other_relevant_information': 2,
    'rescue_volunteering_or_donation_effort': 3,
    'vehicle_damage': 4,
    'affected_individuals': 5,
    'injured_or_dead_people': 5,
    'missing_or_found_people': 5,
}

labels_task3 = {
    'little_or_no_damage': 0,
    'mild_damage': 1,
    'severe_damage' : 2
}

class CrisisMMDataset(BaseDataset):

    def read_data(self, ann_file):
        with open(ann_file, encoding='utf-8') as f:
            self.info = f.readlines()[1:]

        self.data_list = []

        for l in self.info:
            l = l.rstrip('\n')
            #	Generate final_text first from the wiki.py. Check the code of wiki.py and run according to the dataset.

            event_name, tweet_id, image_id, tweet_text,	image,	label,	label_text, label_image, label_text_image, final_text = l.split(
                '\t')

            if self.consistent_only and label_text != label_image:
                continue
            #print(trigger_words)
            self.data_list.append(
                {
                    'path_image': '%s/%s' % (self.dataset_root, image),

                    'text': final_text,
                    'text_tokens': self.tokenize(final_text),

                    'label_str': label,
                    'label': self.label_map[label],

                    'label_image_str': label,
                    'label_image': self.label_map[label],

                    'label_text_str': label,
                    'label_text': self.label_map[label]
                }
            )

    def tokenize(self, sentence):
        ids = self.tokenizer(clean_text(
            sentence), padding='max_length', max_length=512, truncation=True).items()
        return {k: torch.tensor(v) for k, v in ids}

    def initialize(self, opt, phase='train', cat='all', task='task2', shuffle=False, consistent_only=False):
        self.opt = opt
        self.shuffle = shuffle
        self.consistent_only = consistent_only
        self.dataset_root = f'{dataroot}/CrisisMMD_v2.0' if opt.debug else f'{dataroot}/CrisisMMD_v2.0'
        self.image_root = f'{self.dataset_root}/data_image'
        self.label_map = None
        if task == 'task1':
            self.label_map = labels_task1
        elif task == 'task2_full':
            self.label_map = labels_task2_full
        elif task == 'task2':
            self.label_map = labels_task2
        elif task == 'task3':
            self.label_map = labels_task3

        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
        self.tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
        #self.tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        ann_file = '%s/crisismmd_datasplit_all/task_%s_text_img_%s.tsv' % (
            self.dataset_root, task_dict[task], phase
        )

        # Append list of data to self.data_list
        self.read_data(ann_file)

        if self.shuffle:
            np.random.default_rng(seed=0).shuffle(self.data_list)
        self.data_list = self.data_list[:self.opt.max_dataset_size]
        cprint('[*] %d samples loaded.' % (len(self.data_list)), 'yellow')

        self.N = len(self.data_list)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        self.transforms = transforms.Compose([
            transforms.Lambda(lambda img: expand2square(img)),
            transforms.Resize((opt.load_size, opt.load_size)),
            transforms.RandomHorizontalFlip(0.2),
            transforms.RandomCrop((opt.crop_size, opt.crop_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, index):
        data = self.data_list[index]

        to_return = {}
        for k, v in data.items():
            to_return[k] = v

        with Image.open(data['path_image']).convert('RGB') as img:
            image = self.transforms(img)
        to_return['image'] = image
        return to_return

    def __len__(self):
        return len(self.data_list)

    def name(self):
        return 'CrisisMMDataset'


if __name__ == '__main__':

    opt = object()

    dset = CrisisMMDataset(opt, 'train')
    import pdb
    pdb.set_trace()
