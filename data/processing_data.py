import librosa
import numpy as np
import glob
import os
import torch
import random
from sklearn import preprocessing  

# Import parent path
import sys
sys.path.append("..")
from utils import augmentations
from utils import log, data_generator_cls, show
import warnings
warnings.filterwarnings("ignore")
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def process():

    # set list according to features
    mfccs = []
    train_labels = []

    # count each kind of human's number
    copd_path_list = []
    healthy_path_list = []
    raw_data_path = "./raw_data"
    copd_human_count = 0
    healthy_human_count = 0
    for i in glob.glob(os.path.join(raw_data_path,"COPD","*")):
        copd_human_count = copd_human_count + 1
        copd_path_list.append(i)
    for i in glob.glob(os.path.join(raw_data_path,"Healthy","*")):
        healthy_human_count = healthy_human_count + 1
        healthy_path_list.append(i)

    # set shuffled index list
    random.shuffle(copd_path_list)
    random.shuffle(healthy_path_list)

    # 3/5 for train, 1/5 for valid, 1/5 for test
    # train-copd
    print("copd_human_count",copd_human_count)
    print("copd_human_count * 4//5:",copd_human_count * 4//5)
    set_data_list(1, 0, copd_human_count * 4//5,  copd_path_list, mfccs, train_labels)
    # train-healthy
    set_data_list(0, 0, healthy_human_count* 4//5,  healthy_path_list, mfccs, train_labels)    
    # save as train.pt
    save_pt("./processed_data/train.pt", mfccs, train_labels)

    # test-copd
    set_data_list(1, copd_human_count * 4//5, copd_human_count, copd_path_list, mfccs, train_labels, "raw")
    # test-healthy
    set_data_list(0, healthy_human_count * 4//5, healthy_human_count, healthy_path_list, mfccs, train_labels, "raw")
    # save as valid.pt
    save_pt("./processed_data/test.pt", mfccs, train_labels)


# build a dictionary as the following type:
# [] represents a list
# {"mfccs":mfccs, "train_labels":train_labels}
def save_pt(name, mfccs, train_labels):
    shuffle_index = np.random.permutation(np.arange(len(train_labels)))
    feature_dict = {
    "mfccs": np.array(mfccs)[shuffle_index],
    "train_labels": np.array(train_labels)[shuffle_index],
    }
    
    torch.save(feature_dict, name)
    mfccs.clear()
    train_labels.clear()


def set_data_list(label, start, end, name_list, mfccs, train_labels, switch="aug"):
    if switch == "raw":
        for path_num in range(start, end):
            dir_path = name_list[path_num]
            for wav_path in glob.glob(os.path.join(dir_path,"*")):
                mfcc, sound, sample_rate = get_features(wav_path)
                train_label = label
                mfccs.append(mfcc)
                # train_labels==[0,0,0,0,1,1,1,1,....]
                train_labels += [train_label]
    else:
        if label == 0:
            for path_num in range(start, end):
                dir_path = name_list[path_num]
                for wav_path in glob.glob(os.path.join(dir_path,"*")):
                    mfcc, sound, sample_rate = get_features(wav_path)
                    train_label = label
                    augs = augmentations.Augmentation(mfcc, sound, sample_rate)
                    augs_list = augs.aug()
                    mfccs.extend(augs_list)
                    # train_labels==[0,0,0,0,1,1,1,1,....]
                    train_labels += 32*[train_label]
        else:
            for path_num in range(start, end):
                dir_path = name_list[path_num]
                for wav_path in glob.glob(os.path.join(dir_path,"*")):
                    mfcc, sound, sample_rate = get_features(wav_path)
                    train_label = label
                    n = 3
                    for _ in np.arange(n):
                        augs = augmentations.Augmentation(mfcc, sound, sample_rate)
                        augs_list = augs.aug()
                        mfccs.extend(augs_list)
                        # train_labels==[0,0,0,0,1,1,1,1,....]
                        train_labels += 32*[train_label]
                        # mfccs += n*[mfcc]
                        # # train_labels==[0,0,0,0,1,1,1,1,....]
                        # train_labels += n*[train_label]


def get_features(audio_path):
    sound, sample_rate = librosa.load(audio_path)
    # delete zeros at the begining or the end
    sound = np.trim_zeros(sound)
    # Short-time Fourier transform
    stft = np.abs(librosa.stft(sound))
    # mfcc：Return shape=[Number of mfcc, ceil(frame length/frame shift) (default frame shift is 1024)]
    mfcc = librosa.feature.mfcc(y=sound, sr=sample_rate, n_mfcc=16)
    # Spectral contrast：Return shape=[Number of bands +1 (default is 6+1=7), ceil(frame length/frame shift)]
    mfcc= augmentations.trimming_feature(mfcc)
    # Normalization
    mfcc = normalization(mfcc)
    return mfcc, sound, sample_rate

def normalization(input):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    normed = min_max_scaler.fit_transform(input)
    return normed
        
if __name__ == "__main__":
    # set seed
    setup_seed(42)
    process()
