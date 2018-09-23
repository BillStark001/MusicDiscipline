# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:36:25 2018

@author: BillStark001
"""

import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

save_path = 'F:/Datasets/fake_maps/'
load_path = 'data/midi_orig/'
load_sch = load_path + 'sch.pkl'
sch = []
sch_gen = []
with open(load_sch, 'rb') as f:
    sch = pickle.load(f)

def gen_file_names(s, category=8):
    midi_name = save_path + 'midi/{}_midi.bmp'.format(s)
    wave_name = []
    wave_comp_name = []
    for i in range(category):
        wave_name.append(save_path + 'wave/{}_wave_{:0>4}.bmp'.format(s, i))
        wave_comp_name.append(save_path + 'wave/{}_wave_comp_{:0>4}.bmp'.format(s, i))
    return midi_name, wave_name, wave_comp_name

for i in sch:
    sch_gen.append(gen_file_names(i))
    
midi = np.zeros((128, 0))
wave = np.zeros((256, 0))
comp = np.zeros((256, 0))
#m_temp, w_temp, c_temp = 0, 0, 0
point = 0
    
def read_data(f=240, i=0, r=False):
    global midi, wave, comp
    #global m_temp, w_temp, c_temp
    m_temp = cv2.imread(sch_gen[f][0], cv2.IMREAD_GRAYSCALE)
    w_temp = cv2.imread(sch_gen[f][1][i], cv2.IMREAD_GRAYSCALE)
    c_temp = cv2.imread(sch_gen[f][2][i], cv2.IMREAD_GRAYSCALE)
    #c_temp = cv2.resize(c_temp, (int(c_temp.shape[1] * 2.5), 256))
    #print(midi.shape, m_temp.shape)
    #print(m_temp.shape, w_temp.shape, c_temp.shape)
    sp = min(m_temp.shape[1], w_temp.shape[1])
    sp = min(sp, int(c_temp.shape[1]*2.5))
    m_temp = cv2.resize(m_temp, (sp, 128))
    w_temp = cv2.resize(w_temp, (sp, 256))
    c_temp = cv2.resize(c_temp, (sp, 256))
    if r: return m_temp, w_temp, c_temp
    #print(midi.shape, m_temp.shape)
    midi = np.concatenate((midi, m_temp), axis=1)
    wave = np.concatenate((wave, w_temp), axis=1)
    comp = np.concatenate((comp, c_temp), axis=1)
    #print(midi.shape, m_temp.shape)

#m, w, c = read_data(r=True)

def gen_data(length=25, batch_size=32):
    #print('g')
    while True:
        global midi, wave, comp, point
        x = []
        y = []
        for i in range(batch_size):
            if point >= midi.shape[1] - length:
                midi = midi[:, point:]
                wave = wave[:, point:]
                comp = comp[:, point:]
                read_data(f=np.random.randint(len(sch)), i=np.random.randint(8))
                point = 0
            x.append(np.array((wave[:, point: point + length], comp[:, point: point + length])).reshape(512, length).transpose(1, 0))
            y.append(midi[:, point: point + length].transpose(1, 0))
            point += length
            #print('kai: point=%d, i=%d'%(point, i))
        x = np.array(x)
        y = np.array(y)
        #print('kang: point=%d'%point)
        yield x, y
    
if __name__ == '__main__':
    g = gen_data()
    while True:
        x, y = next(g)