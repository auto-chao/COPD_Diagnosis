import numpy as np
import librosa
from sklearn import preprocessing  

class Augmentation():
    def __init__(self, mfcc, sound, sample_rate) -> None:
        self.sample_rate = sample_rate
        self.mfcc = mfcc
        self.sound = sound

    def aug(self):
        result_list = []
        method_list = [self.volume_augment, self.time_stretch, self.time_shift, self.n_mfcc_shift,  self.jitter]
        input_list = [self.sound]
        output_list = []
        for i in range(len(method_list)):
            for input in input_list:
                if method_list[i] == self.volume_augment:
                    add = method_list[i](input)
                    no_add = self.mfcc
                else:
                    add = method_list[i](input)
                    no_add = input
                output_list.extend([add, no_add])
            input_list.clear()
            input_list = output_list.copy()
            if i != len(method_list) - 1:
                output_list.clear()
        return output_list
        
    def jitter(self, input, sigma=0.8):
        mfcc = input.copy()
        return mfcc + np.random.normal(loc=0., scale=sigma, size=mfcc.shape)

    def n_mfcc_shift(self, input):
        """ Shift a mfcc along the n_mfcc axis in the spectral-domain at random"""
        mfcc = input.copy()
        nb_cols = mfcc.shape[1]
        mamfcc_shifts = nb_cols//20 # around 5% shift
        nb_shifts = np.random.randint(-mamfcc_shifts, mamfcc_shifts)
        return np.roll(mfcc, nb_shifts, axis=0)
    
    def time_shift(self, input):
        mfcc = input.copy()
        nb_cols = mfcc.shape[0]
        nb_shifts = np.random.randint(0, nb_cols)
        return np.roll(mfcc, nb_shifts, axis=1)

    def time_stretch(self, input):
        mfcc = input.copy()
        rate = randfloat(1, 0.8, 1.5) # rate is 0.8~1.5
        return trimming_feature(librosa.effects.time_stretch(mfcc, rate))
    
    def volume_augment(self, input, min_gain_dBFS=-10, mamfcc_gain_dBFS=10): # 0.316~3.16
        sound = input.copy()  
        data_type = sound[0].dtype
        gain = np.random.uniform(min_gain_dBFS, mamfcc_gain_dBFS)
        gain = 10. ** (gain / 20.)
        sound = sound * gain
        sound = sound.astype(data_type)
        mfcc = librosa.feature.mfcc(y=sound, sr=self.sample_rate, n_mfcc=16)
        mfcc = normalization(mfcc)
        return trimming_feature(mfcc)


def normalization(input):
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
    normed = min_max_scaler.fit_transform(input)
    return normed


def trimming_feature(*args):
    for _ in args:
        if len(_.shape) < 2:
            if _.shape[0] <= 512:
                _ = np.pad(_, pad_width= (0, 512 - _.shape[0]))
            else:
                _ = _[:512]
        else:
            _ = np.pad(_, pad_width=((0, 16 - _.shape[0]), (0, 0)))
            if _.shape[1] <= 512:
                _ = np.pad(_, pad_width=((0, 0), (0, 512 - _.shape[1])))
            else:
                _ = _[:, :512]
    return _

def randfloat(num, l, h):
    if l > h:
        return None
    else:
        a = h - l
        b = h - a
        out = (np.random.rand(num) * a + b).item()
        return out
 


