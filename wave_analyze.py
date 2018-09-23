import wave
import numpy as np
import math
import cv2
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

def calc_line(x1, x2, y1, y2):
    k = (y2 - y1) / (x2 - x1)
    if x1 == x2: k = 0
    b = y1 - k * x1
    return k, b
    
def sample_curve(x, y, smin=0, smax=128, d=0.5, subsmpl=4, zero_division=1e-8, m_threshold = 1):
    x = np.array(x)
    y = np.array(y)
    smpl = math.ceil((smax - smin) / d)
    ans = np.zeros(smpl)
    if x.size == 0: return ans
    subs = []
    sup = len(x)
    inf = len(x)
    for i in range(len(x)): 
        if x[i] >= smin: 
            inf = i
            break
    for i in range(len(x)): 
        if x[i] >= smax: 
            sup = i
            break
    if inf == 0:
        #if x[inf] == smin: x[inf] += zero_division
        if x[0] - smin > m_threshold:
            x = np.concatenate(([x[0] - m_threshold], x))
            y = np.concatenate(([0], y))
            #inf += 1
            sup += 1
        
        x = np.concatenate(([smin], x))
        y = np.concatenate(([0], y))
        inf += 1
        sup += 1
    if sup == len(x):
        #if x[sup] == smax: x[sup] -= zero_division
        
        if x[-1] + m_threshold < smax:
            x = np.concatenate((x, [x[-1] + m_threshold]))
            y = np.concatenate((y, [0]))
            sup += 1
        
        x = np.concatenate((x, [smax]))
        y = np.concatenate((y, [0]))
    x = x[inf - 1: sup + 1]
    y = y[inf - 1: sup + 1]
    #for i, j in zip(x, y):
    #    print(i, j)
    x_cur = 1
    sub_cur = 0
    k_cur, b_cur = calc_line(x[x_cur - 1], x[x_cur], y[x_cur - 1], y[x_cur])
    for i in np.arange(smin, smax, d):#/subsmpl):
        #print(i)
        while i > x[x_cur]: 
            x_cur += 1
            k_cur = (y[x_cur] - y[x_cur - 1]) / (x[x_cur] - x[x_cur - 1])
            if x[x_cur] == x[x_cur - 1]: k_cur = 0
            b_cur = y[x_cur - 1] - k_cur * x[x_cur - 1]
        subs = k_cur * i + b_cur
        #subs.append(k_cur * i + b_cur)
        #if len(subs) == subsmpl:
        ans[sub_cur] = subs#np.mean(np.array(subs))
        sub_cur += 1
        #    subs = 0#[]
    '''
    #for i in x: print(i)
    for i in range(len(x)-1):
        x_cur, y_cur = x[i], y[i]
        x_pos = x_cur / d
        x_sub = x_pos % 1
        x_pos = int(x_pos - x_sub)
        #print(x_pos, x_sub)
        ans[x_pos] += y_cur * (1 - x_sub)
        ans[x_pos + 1] += y_cur * x_sub
    '''
    return ans
    
def timewise_fft(wave, enc=44100, sep=441, smpls=256):
    ans = np.zeros((smpls, math.ceil(len(wave) / sep)))
    for i in tqdm(range(ans.shape[1]), desc='fft  ', ncols=64):
        wave_cur = wave[sep * i: sep * (i + 1)] 
        fft_y, fft_x = wavfft(wave_cur, enc)
        smpl = sample_curve(fft_x, fft_y, smin=0, smax=128, d=128/smpls, subsmpl=1).reshape((smpls, 1))
        #plt.plot(np.arange(smpl.size) / 2, smpl, linewidth=0.7)
        #plt.show()
        ans[:, i] = smpl[:, 0]
    return ans

def get_fft_map(path, reader=read_mp3, channel=0, sep=441):
    wavdata, params = reader(path)
    fft_map = timewise_fft(wavdata[0][:-2], enc=params['frame_rate'], sep=sep)
    return fft_map
    
if __name__ == '__main__':
    #c = np.arange(0, 128, 2)
    #c = np.concatenate((c[:20], c[-20:]))
    #d = np.random.random(c.shape)
    #a = sample_curve(c, d)
    path = 'data/midi_test/1.mp3'
    fft_map = get_fft_map(path, sep=4410)
    plt.imshow(fft_map[:, :500]); plt.show()
    fft_map = fft_map[:, :1000]
    cv2.imwrite('441.bmp', fft_map / (2500 * 5))
