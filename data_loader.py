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
point = 0
    
def read_data(f=50, i=0, r=False):
    global midi, wave, comp
    m_temp = cv2.imread(sch_gen[f][0], cv2.IMREAD_GRAYSCALE)
    w_temp = cv2.imread(sch_gen[f][1][i], cv2.IMREAD_GRAYSCALE)
    c_temp = cv2.imread(sch_gen[f][2][i], cv2.IMREAD_GRAYSCALE)
    c_temp = cv2.resize(c_temp, (int(c_temp.shape[1] * 2.5), 256))
    #print(m_temp.shape, w_temp.shape, c_temp.shape)
    sp = min(m_temp.shape[1], w_temp.shape[1])
    sp = min(sp, c_temp.shape[1])
    if r: return m_temp[:, :sp], w_temp[:, :sp], c_temp[:, :sp]
    midi = np.concatenate((midi, m_temp[:, :sp]), axis=1)
    wave = np.concatenate((wave, w_temp[:, :sp]), axis=1)
    comp = np.concatenate((comp, c_temp[:, :sp]), axis=1)

#read_data()

def gen_data(length=25, batch_size=32):
    global midi, wave, comp, point
    x = []
    y = []
    for i in range(batch_size):
        if point >= midi.shape[1] - length:
            midi = midi[point:]
            wave = wave[point:]
            comp = comp[point:]
            read_data()#(f=np.random.randint(len(sch)), i=np.random.randint(8))
            point = 0
        x.append(np.array((wave[:, point: point + length], comp[:, point: point + length])).reshape(length, 256, 2))
        y.append(midi[:, point: point + length].reshape(length, 128))
        point += length
    yield x, y
    