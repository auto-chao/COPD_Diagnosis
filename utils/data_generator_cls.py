import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
import librosa
def data_generator(dataset, args):
    dataset = Load_Dataset(dataset, args)
    # drop_last: defaults=False, means if the last batch contains less than one batch_size, the batch is reserved
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)
    return loader

class Load_Dataset(Dataset):
    # need to overwrite the 
    def __init__(self, dataset, args):
        super(Load_Dataset, self).__init__()
        # Parse out the respective data and labels
        self.feature = torch.from_numpy(dataset["mfccs"]).float().transpose(1, 2)
        self.label = torch.from_numpy(dataset['train_labels']).long()
        self.len = self.feature.shape[0]
    
    def __getitem__(self, index):
        return self.feature[index], self.label[index]
    
    def __len__(self):
        return self.len