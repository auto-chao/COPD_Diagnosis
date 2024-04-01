import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torchaudio
import torch.nn.functional as F


def data_generator(dataset, args):
    dataset = Load_Dataset(dataset, args)
    # drop_last: defaults=False, means if the last batch contains less than one batch_size, the batch is reserved
    loader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                        shuffle=True, drop_last=False, num_workers=args.num_workers)
    return loader


class Load_Dataset(Dataset):
    # need to overwrite the
    def __init__(self, dataset, args):
        super(Load_Dataset, self).__init__()
        mfcc_GPUtensor = torch.from_numpy(dataset["mfccs"]).float().transpose(1, 2).to(args.device)
        mfcc_GPUtensor_add_noise = mfcc_GPUtensor + torch.randn(mfcc_GPUtensor.shape).to(args.device)
        # Parse out the respective data and labels
        freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=6)
        time_masking = torchaudio.transforms.TimeMasking(time_mask_param=64, p=0.4)
        self.aug1 = time_masking(freq_masking(mfcc_GPUtensor_add_noise))
        self.aug2 = time_masking(freq_masking(mfcc_GPUtensor))

        self.len = self.aug1.shape[0]

    def __getitem__(self, index):
        return self.aug1[index], self.aug2[index]

    def __len__(self):
        return self.len
