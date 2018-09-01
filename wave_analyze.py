import wave
import numpy as np
import math
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from pydub import AudioSegment
from tqdm import tqdm

def read_wav(path):
    f = wave.open(path, "rb")
    params = f.getparams()
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)
    wave_data.shape = -1, 4
    wave_data = wave_data.T
    wave_data = np.array([wave_data[1], wave_data[3]])
    params = {
              'frame_rate': framerate, 
              'channels': nchannels
             }
    return wave_data, params
    
def read_mp3(path):
    wave_orig = AudioSegment.from_mp3(path)
    wave = wave_orig.get_array_of_samples()
    wave = np.array(wave, dtype='int16')
    wave.shape = -1, wave_orig.channels
    wave = wave.T
    params = {
              'frame_rate': wave_orig.frame_rate, 
              'channels': wave_orig.channels
             }
    return wave, params

def wavfft(wave, sample=44100):
    try:
        fft_temp = abs(fft(wave))
    except:
        fft_temp = np.zeros(128)
    fft_result=fft_temp[range(int(len(fft_temp)/2))]
    h=np.arange(len(fft_result))/len(wave)*sample
    note=np.log(h)/np.log(np.power(2, 1/12))-36.5
    cut_beg, cut_end=0, 0
    for i in range(len(note)):
        if(note[i]>=0):
            cut_beg=i;break
    for i in range(len(note)):
        if(note[i]>=96):
            cut_end=i;break
    return fft_result[cut_beg:cut_end], note[cut_beg:cut_end]

def int2note(x):
    letter=('C', 'C', 'D', 'D', 'E', 'F', 'F', 'G', 'G', 'A', 'A', 'B')
    ascend=('', '#', '', '#', '', '', '#', '', '#', '', '#', '')
    x1, x2=math.floor(x/12), int(x%12)
    res=letter[x2]+str(x1)+ascend[x2]
    return res
    
def sample_curve(x, y, smpl=256, d=0.5, d_min = 0.005):
    ans = np.zeros(smpl)
    #for i in x: print(i)
    for i in range(len(x)-1):
        x_cur, y_cur = x[i], y[i]
        x_pos = x_cur / d
        x_sub = x_pos % 1
        x_pos = int(x_pos - x_sub)
        #print(x_pos, x_sub)
        ans[x_pos] += y_cur * (1 - x_sub)
        ans[x_pos + 1] += y_cur * x_sub
    return ans
    
def timewise_fft(wave, enc=44100, sep=441, smpls=256):
    ans = np.zeros((smpls, math.ceil(len(wave) / sep)))
    for i in tqdm(range(ans.shape[1])):
        wave_cur = wave[sep * i: sep * (i + 1)] 
        fft_y, fft_x = wavfft(wave_cur, enc)
        smpl = sample_curve(fft_x, fft_y, smpl=smpls, d=128/smpls).reshape((smpls, 1))
        #plt.plot(np.arange(smpl.size) / 2, smpl, linewidth=0.7)
        #plt.show()
        ans[:, i] = smpl[:, 0]
    return ans

def get_fft_map(path, reader=read_mp3, channel=0):
    wavdata, params = reader(path)
    fft_map = timewise_fft(wavdata[0][:-2], enc=params['frame_rate'])
    return fft_map
    
if __name__ == '__main__':
    path = 'data/midi_test/1.mp3'
    fft_map = get_fft_map(path)
    plt.imshow(fft_map[:, :500]); plt.show()
