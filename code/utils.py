import soundfile as sf
import numpy as np

speaker_path = '../dataset/speaker_signal_16000.wav'
mic_path = '../dataset/recorded_signal_16000_trim.wav'

def load_data(mode='speaker'):
    if mode=='speaker':
        data, sr = sf.read(speaker_path)
    elif mode=='mic':
        data, sr = sf.read(mic_path)
    return data, sr

def write_data(data_name, data, sample_rate):
    return sf.write(data_name, data, sample_rate)

if __name__ == '__main__':
    spk_data, spk_sr = load_data(mode='speaker')
    mic_data, mic_sr = load_data(mode='mic')    
    #info of spk_data
    #print(spk_data.dtype) #float64
    #print(spk_data.shape) #127594
    #print(spk_sr) #16000
    
    #info of mic_data
    #print(mic_data.dtype) #float64
    #print(mic_data.shape) #127594
    #print(mic_sr) #16000
