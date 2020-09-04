import numpy as np
import librosa
import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set()
import os
import scipy
import librosa.display
import random

audio_file_path=r'./test/1/1.wav'

sample_rate, samples = scipy.io.wavfile.read(audio_file_path)

samples,sample_rate=librosa.load(audio_file_path,sr = None,mono=True)
# print(samples.shape)
# plt.figure(figsize=(12, 4))
# librosa.display.waveplot(samples.astype('float'), sr=sample_rate)
# plt.show()

X = librosa.stft(samples.astype('float'))
Xdb = librosa.amplitude_to_db(X)
# plt.figure(figsize=(12, 5))
# librosa.display.specshow(Xdb, sr=sample_rate, x_axis='time', y_axis='hz')
# plt.show()

def change_pitch_and_speed(samples):

    y_pitch_speed = samples.copy()
    # you can change low and high here
    length_change = np.random.uniform(low=0.8, high = 1.2)
    speed_fac = 1.0  / length_change
    print("resample length_change = ",length_change)
    tmp = np.interp(np.arange(0,len(y_pitch_speed),speed_fac),np.arange(0,len(y_pitch_speed)),y_pitch_speed)  #通过线性插值法得到改速后的音频数组

    minlen = min(y_pitch_speed.shape[0], tmp.shape[0])
    y_pitch_speed *= 0
    y_pitch_speed[0:minlen] = tmp[0:minlen]


    return y_pitch_speed

def change_pitch_only(samples,sample_rate):
    y_pitch = samples.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change = pitch_pm * 2 * (np.random.uniform())
    print("pitch_change = ", pitch_change)
    y_pitch = librosa.effects.pitch_shift(y_pitch.astype('float64'),
                                          sample_rate, n_steps=pitch_change,
                                          bins_per_octave=bins_per_octave)

    return y_pitch


def change_speed_only(samples):
    y_speed = samples.copy()
    speed_change = np.random.uniform(low=0.9, high=1.1)
    print("speed_change = ", speed_change)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0
    y_speed[0:minlen] = tmp[0:minlen]

    return y_speed

def value_augmentation(samples):
    y_aug = samples.copy()
    dyn_change = np.random.uniform(low=1.5, high=3)
    print("dyn_change = ", dyn_change)
    y_aug = y_aug * dyn_change
    print(samples[:10])
    print(y_aug[:10])

    return y_aug

def add_distribution_noise(samples):
    y_noise = samples.copy()
    # you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005 * np.random.uniform() * np.amax(y_noise)
    y_noise = y_noise.astype('float64') + noise_amp * np.random.normal(size=y_noise.shape[0])

    return y_noise

def random_shifting(samples):
    y_shift = samples.copy()
    timeshift_fac = 0.2 * 2 * (np.random.uniform() - 0.5)  # up to 20% of length
    print("timeshift_fac = ", timeshift_fac)
    start = int(y_shift.shape[0] * timeshift_fac)
    print(start)
    if (start > 0):
        y_shift = np.pad(y_shift, (start, 0), mode='constant')[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift, (0, -start), mode='constant')[0:y_shift.shape[0]]

    return y_shift


def apply_hpss(samples):
    y_hpss = librosa.effects.hpss(samples.astype('float64'))
    # print(y_hpss[1][:10])
    # print(samples[:10])
    y_hpss=np.asarray(y_hpss)
    return y_hpss

y=apply_hpss(samples)

print(type(y))

# def shift_silent_to_the_right(samples):
#     sampling = samples[(samples > 200) | (samples < -200)]  #音频中非静音部分
#     shifted_silent = sampling.tolist() + np.zeros((samples.shape[0] - sampling.shape[0])).tolist()
#
#     return shifted_silent


def Streching(samples):
    input_length = len(samples)
    streching = samples.copy()
    streching = librosa.effects.time_stretch(streching.astype('float'), 1.1)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")

    return streching



def audio_augment(samples,sample_rate):

    seed=random.randint(0,8)

    if seed==0:
        rt_samples=change_pitch_and_speed(samples)
    elif seed==1:
        rt_samples=change_pitch_only(samples,sample_rate)
    elif seed==2:
        rt_samples=change_speed_only(samples)
    elif seed==3:
        rt_samples=value_augmentation(samples)
    elif seed==4:
        rt_samples=add_distribution_noise(samples)
    elif seed==5:
        rt_samples =random_shifting(samples)
    elif seed==6:
        rt_samples=apply_hpss(samples)
    else:
        rt_samples=Streching(samples)

    return rt_samples



if __name__=='__main__':
    samples, sample_rate = librosa.load(audio_file_path, sr=None, mono=True)
    samples=audio_augment(samples,sample_rate)


    save_path=r'./test/audio_augment.wav'

    librosa.output.write_wav(save_path, samples, sample_rate, norm=True)
