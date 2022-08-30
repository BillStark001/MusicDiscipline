import os
import wave
import numpy as np
import math
import cv2
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf
from collections import defaultdict

'''
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
    
from pydub import AudioSegment
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
'''

import librosa
def read_file(path, sr=44100):
    y, sr = librosa.load(path, sr=sr)
    return [y], dict(frame_rate=sr)
    

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

def get_fft_map(path, sr=44100, channel=0, sep=441):
    wavdata, params = read_file(path, sr=sr)
    print(np.shape(wavdata), params)
    fft_map = timewise_fft(wavdata[0][:-2], enc=params['frame_rate'], sep=sep)
    return fft_map

def sep_d53(fft_map, time_scale_f = 44100 / 4410, low_detach_t = 1, d5_t = (150, 154), d3_t = (143, 147), 
            d5_low_duration_t = 4, d3_low_duration_t = 12, d53_error_f = 1):
    
    fa = np.argmax(fft_map, axis=0)
    fm = np.max(fft_map, 0)
    fa[fm < low_detach_t] = 1
    
    fa_d5 = np.r_[np.logical_and(fa > d5_t[0], fa < d5_t[1]), np.array([False, False, False])]
    fa_d3 = np.r_[np.logical_and(fa > d3_t[0], fa < d3_t[1]), np.array([False, False, False])]
    fa_d5[1:] = np.logical_xor(fa_d5[1:], fa_d5[:-1])
    fa_d3[1:] = np.logical_xor(fa_d3[1:], fa_d3[:-1])
    fa_d5_where = np.where(fa_d5)[0].reshape((-1, 2))
    fa_d3_where = np.where(fa_d3)[0].reshape((-1, 2))
    fa_d5_where[..., 1] -= fa_d5_where[..., 0]
    fa_d3_where[..., 1] -= fa_d3_where[..., 0]
    fa_d5_pairs = [x for x in fa_d5_where if x[1] > d5_low_duration_t]
    fa_d3_pairs = [x for x in fa_d3_where if x[1] > d3_low_duration_t]
    
    d5_list = [x[0] / time_scale_f for x in fa_d5_pairs if x[1] > d3_low_duration_t]
    d3_list = [x[0] / time_scale_f for x in fa_d3_pairs]
    d53_tmp_list = [x[0] / time_scale_f for x in fa_d5_pairs if x[1] <= d3_low_duration_t]
    d53_list = []
    d3_mask = [False] * len(d3_list)
    for i in range(len(d3_list)):
        judge_arr = [x for x in d53_tmp_list if x < d3_list[i] and np.abs(x-d3_list[i]) <= d53_error_f]
        d3_mask[i] = len(judge_arr) > 0
        if d3_mask[i]:
            d53_list.append(judge_arr[0])
    d3_list = [d3_list[i] for i in range(len(d3_list)) if not d3_mask[i]]
    
    return d5_list, d3_list, d53_list
    
src_path = 'G:/EJU_Original/听力/'
dst_path = 'G:/EJU_Original/Listening_Decomp/{testtime}/'
dst_pattern = dst_path + '{testtime}_{segno}.wav'

src_lst = os.listdir(src_path)

mapping_list = defaultdict(lambda: 'gshxd', {
    (30, 14, 6): defaultdict(lambda: -1, 
                             dict(np.array([[2 * (x + 4) for x in range(12)] + [x+ 35 for x in range(15)], list( range(1, 28) ) ]).T)
                             ),
    (27, 12, 0): defaultdict(lambda: -1, 
                             dict(np.array([[2 * (x + 0) + 1 for x in range(12)] + [x+ 25 for x in range(15)], list( range(1, 28) ) ]).T)
                             )
    })

if __name__ == '__main__':

    filename = src_lst[-9]
    path = src_path + filename
    testtime = filename.strip('.m4a')
    sr = 44100
    sep = 4410
    space_correction_t = 0.4
    wavdata, params = read_file(path, sr=sr)
    wavdata = wavdata[0]
    fft_map = timewise_fft(wavdata, enc=sr, sep=sep)
    sep_res = sep_d53(fft_map, time_scale_f=sr/sep)
    merged_segs = sorted([[int((x - space_correction_t) * sr), 'd5'] for x in sep_res[0]] \
                         + [[int((x - space_correction_t) * sr), 'd3'] for x in sep_res[1]] \
                         + [[int((x - space_correction_t) * sr), 'd53'] for x in sep_res[2]] \
        , key=lambda x: x[0])
        
    try:
        os.makedirs(dst_path.format(testtime=testtime))
    except Exception as e:
        print(e)
    
    cur_list = mapping_list[tuple([len(x) for x in sep_res])]
    if cur_list == 'gshxd':
        print([len(x) for x in sep_res])
        
    for i in tqdm(range(len(merged_segs) + 1), desc='SGIO'):
        if cur_list == 'gshxd':
            segno = i
        else:
            segno = cur_list[i]
            if segno == -1:
                continue
        cur_time = [0, 'dd'] if i == 0 else merged_segs[i - 1]
        nxt_time = [wavdata.size, 'dd'] if i == len(merged_segs) else merged_segs[i]
        print(cur_time, nxt_time)
        sav_path = dst_pattern.format(testtime=testtime, segno=segno)
        sf.write(sav_path, np.vstack([wavdata[max(0, cur_time[0]): max(0, nxt_time[0])]] * 2).T, samplerate=44100)


