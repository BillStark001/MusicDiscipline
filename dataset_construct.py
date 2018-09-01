# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 20:36:25 2018

@author: BillStark001
"""

import midi_analyze as midi
import wave_analyze as wave

import os
import numpy as np
import cv2
import pickle
from tqdm import tqdm

save_path = 'F:/Datasets/fake_maps/'
midi_save_path = save_path + 'midi/'
temp_save_path = save_path + 'temp/'
wave_save_path = save_path + 'wave/'
schedule_path = save_path + 'records.csv'
load_path = 'F:/Programs/ML/MusicDiscipline/data/midi_orig/'
load_sch = load_path + 'sch.pkl'
sch = []
sch_gen = []
with open(load_sch, 'rb') as f:
    sch = pickle.load(f)

sch = sch[82:]

midi_division = 1#256
fft_division = midi_division * 5000

def gen_file_names(s, category=8):
    midi_name = 'midi/{}_midi.bmp'.format(s)
    midi_temp_name = []
    wave_temp_name = []
    wave_name = []
    for i in range(category):
        midi_temp_name.append('temp/{}_midi_temp_{:0>4}.mid'.format(s, i))
        wave_temp_name.append('temp/{}_midi_temp_{:0>4}.mp3'.format(s, i))
        wave_name.append('wave/{}_wave_{:0>4}.bmp'.format(s, i))
    return midi_name, midi_temp_name, wave_temp_name, wave_name

for i in sch:
    sch_gen.append(gen_file_names(i))
    
for i in tqdm(range(len(sch)), desc='file ', ncols=64):
    #midi_map = midi.get_midi_map(load_path + sch[i] + '.mid')
    #midi_map = midi_map / midi_division
    #cv2.imwrite(save_path + sch_gen[i][0], midi_map)
    
    #for j in range(len(sch_gen[i][1])):
    #    midi.convert_midi_insts_to_piano(load_path + sch[i] + '.mid', save_path + sch_gen[i][1][j], category=j)
    
    for j in tqdm(range(len(sch_gen[i][1])), desc='inst ', ncols=64):
        try:
            wave_map = wave.get_fft_map(save_path + sch_gen[i][2][j])
        except:
            continue
        wave_map = wave_map / fft_division
        cv2.imwrite(save_path + sch_gen[i][3][j], wave_map)