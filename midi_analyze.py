# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:03:15 2018

@author: BillStark001
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

def strb_to_int(strb):
    ans = 0
    for i in strb: 
        ans = ans * 16 + i
    return ans

def uint8_to_byte(x):
    c_map = [
             b'\x00', b'\x01', b'\x02', b'\x03', b'\x04', b'\x05', b'\x06', b'\x07',
             b'\x08', b'\x09', b'\x0a', b'\x0b', b'\x0c', b'\x0d', b'\x0e', b'\x0f',
             b'\x10', b'\x11', b'\x12', b'\x13', b'\x14', b'\x15', b'\x16', b'\x17',
             b'\x18', b'\x19', b'\x1a', b'\x1b', b'\x1c', b'\x1d', b'\x1e', b'\x1f',
             b'\x20', b'\x21', b'\x22', b'\x23', b'\x24', b'\x25', b'\x26', b'\x27',
             b'\x28', b'\x29', b'\x2a', b'\x2b', b'\x2c', b'\x2d', b'\x2e', b'\x2f',
             b'\x30', b'\x31', b'\x32', b'\x33', b'\x34', b'\x35', b'\x36', b'\x37',
             b'\x38', b'\x39', b'\x3a', b'\x3b', b'\x3c', b'\x3d', b'\x3e', b'\x3f',
             b'\x40', b'\x41', b'\x42', b'\x43', b'\x44', b'\x45', b'\x46', b'\x47',
             b'\x48', b'\x49', b'\x4a', b'\x4b', b'\x4c', b'\x4d', b'\x4e', b'\x4f',
             b'\x50', b'\x51', b'\x52', b'\x53', b'\x54', b'\x55', b'\x56', b'\x57',
             b'\x58', b'\x59', b'\x5a', b'\x5b', b'\x5c', b'\x5d', b'\x5e', b'\x5f',
             b'\x60', b'\x61', b'\x62', b'\x63', b'\x64', b'\x65', b'\x66', b'\x67',
             b'\x68', b'\x69', b'\x6a', b'\x6b', b'\x6c', b'\x6d', b'\x6e', b'\x6f',
             b'\x70', b'\x71', b'\x72', b'\x73', b'\x74', b'\x75', b'\x76', b'\x77',
             b'\x78', b'\x79', b'\x7a', b'\x7b', b'\x7c', b'\x7d', b'\x7e', b'\x7f',
             b'\x80', b'\x81', b'\x82', b'\x83', b'\x84', b'\x85', b'\x86', b'\x87',
             b'\x88', b'\x89', b'\x8a', b'\x8b', b'\x8c', b'\x8d', b'\x8e', b'\x8f',
             b'\x90', b'\x91', b'\x92', b'\x93', b'\x94', b'\x95', b'\x96', b'\x97',
             b'\x98', b'\x99', b'\x9a', b'\x9b', b'\x9c', b'\x9d', b'\x9e', b'\x9f',
             b'\xa0', b'\xa1', b'\xa2', b'\xa3', b'\xa4', b'\xa5', b'\xa6', b'\xa7',
             b'\xa8', b'\xa9', b'\xaa', b'\xab', b'\xac', b'\xad', b'\xae', b'\xaf',
             b'\xb0', b'\xb1', b'\xb2', b'\xb3', b'\xb4', b'\xb5', b'\xb6', b'\xb7',
             b'\xb8', b'\xb9', b'\xba', b'\xbb', b'\xbc', b'\xbd', b'\xbe', b'\xbf',
             b'\xc0', b'\xc1', b'\xc2', b'\xc3', b'\xc4', b'\xc5', b'\xc6', b'\xc7',
             b'\xc8', b'\xc9', b'\xca', b'\xcb', b'\xcc', b'\xcd', b'\xce', b'\xcf',
             b'\xd0', b'\xd1', b'\xd2', b'\xd3', b'\xd4', b'\xd5', b'\xd6', b'\xd7',
             b'\xd8', b'\xd9', b'\xda', b'\xdb', b'\xdc', b'\xdd', b'\xde', b'\xdf',
             b'\xe0', b'\xe1', b'\xe2', b'\xe3', b'\xe4', b'\xe5', b'\xe6', b'\xe7',
             b'\xe8', b'\xe9', b'\xea', b'\xeb', b'\xec', b'\xed', b'\xee', b'\xef',
             b'\xf0', b'\xf1', b'\xf2', b'\xf3', b'\xf4', b'\xf5', b'\xf6', b'\xf7',
             b'\xf8', b'\xf9', b'\xfa', b'\xfb', b'\xfc', b'\xfd', b'\xfe', b'\xff'
            ]
    
    if not x in range(256): x = 0
    x = c_map[x]
    return x

def parse_delta_time(strb):
    ans = 0
    count = 0
    for i in strb:
        count += 1
        if i > 128:
            ans = ans * 128 + i - 128
        else:
            ans = ans * 128 + i
            break
    return ans, count, strb[count:]
    
def parse_event(strb):
    category = strb[0]
    if category == 255:
        strb = strb[1:]
        category = 'meta'
        sub = strb[0]
        count, _, strb= parse_delta_time(strb[1:])
        event = strb[:count]
        orig = strb[count:] 
        
    elif category == 240:
        strb = strb[1:]
        category = 'sysex'
        sub = 0
        count, _, strb= parse_delta_time(strb[0:])
        event = strb[:count]
        orig = strb[count:] 
        
    else:
        category = 'midi'
        if len(strb) == 1: strb = strb + strb
        sub = strb[0]
        count = 1
        if sub < 128:
            #for i in strb[:4]: print(i)
            sub = 255
            #print(strb[:2])
            event = (strb[0], strb[1])
            orig = strb[2:]
        else:
            if sub < 192 or sub >= 224:
                event = (strb[1], strb[2])#strb[1:2]
                orig = strb[3:]
            else:
                event = strb[1]
                orig = strb[2:]
        
    return category, sub, count, event, orig

def parse_note(n):
    b = n % 12
    o = int((n - b) / 12)
    return (o, b)

def analyze_event(event, 
                  post_event,#=tuple(0, 'midi', 8, {'channel': 0, 'note': (0, 0), 'strength': 0}), 
                  post_midi_event#=tuple(0, 'midi', 8, {'channel': 0, 'note': (0, 0), 'strength': 0}), 
):
    dtime = event[0]
    cat = event[1]
    sub = event[2]
    event = event[3]
    if cat == 'midi':
        subc = int(sub[0], base=16)
        channel = int(sub[1], base=16)
        if subc < 11:
            event = {
                    'channel': channel,
                    'note': parse_note(event[0]),
                    'strength': event[1]
                    }
        elif subc == 11:
            event = {
                    'channel': channel, 
                    'ctrl': event[0],
                    'arg': event[1]
                    }
        elif subc == 12:
            event = {
                    'channel': channel, 
                    'instrument': event, 
                    'category': math.floor(event / 8),
                    'type': event % 8
                    }
        elif subc == 13:
            event = {
                    'channel': channel,
                    'contact': int(event)
                    }
        elif subc == 14:
            event = {
                    'channel': channel, 
                    'pitch': event[0] * 128 + event[1] * 1
                    }
        elif subc == 15:
            if event[1] == 0: subc = 8
            else: subc = 9
            event = {
                    'channel': post_midi_event[3]['channel'],
                    'note': parse_note(event[0]),
                    'strength': event[1]
                    }
        sub = subc
    elif cat == 'meta':
        if sub[0] == '5':
            if sub[1] == '1':
                event = {'dtime': event[0] * 65536 + event[1] * 256 + event[2] * 1}
            elif sub[1] == '8':
                event = {
                         'sort': (event[0], 2 ** event[1]), 
                         'tick': event[2], 
                         'crotchet_contains_32': event[3]
                        }
            elif sub[1] == '9':
                event = {
                         'lift': event[0], 
                         'isminor': event[1]
                        }
            elif sub[1] == '4':
                event = {
                         'hour': event[0], 
                         'minute': event[1], 
                         'second': event[2], 
                         'frame': event[3],
                         'subframe': event[4]
                        }
            sub = 5
        elif sub == '0':
            sub = 5
            event = {'sequence': (event[0], event[1])}
        elif len(sub) == 1:
            m = {
                 '1': 'remark',
                 '2': 'copyright',
                 '3': 'title',
                 '4': 'instrument',
                 '5': 'lyric',
                 '6': 'marker',
                 '7': 'starting_point'
                }
            event = {m[sub]: event.decode(encoding='utf-8')}
            sub = 0
        elif sub[0] == '2':
            if sub[1] == '0':
                event = {'midi_channel': event[0]}
            elif sub[1] == '1':
                event = {'midi_interface': event[0]}
            elif sub[1] == 'f':
                event = {'end_track': 1}
            sub = 2
    elif cat == 'sysex':
        sub = 0
    #print(post_event[0])
    return (dtime + post_event[0], cat, sub, event)

def analyze_track(track, max_index=0, analyze_events=False):
    #for i in track: print(hex(i))
    events = []
    post_event = (0, 'midi', 8, {'channel': 0, 'note': (4, 0), 'strength': 0})
    post_midi_event = (0, 'midi', 8, {'channel': 0, 'note': (4, 0), 'strength': 0})

    c = 0
    while len(track) > 1:
        c += 1
        if c == max_index: break
        dtime, count, track = parse_delta_time(track)
        cat, sub, count, event, track = parse_event(track)
        cur = (dtime, cat, hex(sub)[2:], event)
        if analyze_events:cur = analyze_event(cur, post_event, post_midi_event)
        
        post_event = cur
        if cur[1] == 'midi': post_midi_event = cur
        
        events.append(cur)
        
        #print(cur)
    #if events[-1][2] == 47 or events[-1][2] == '2f': events = events[:-1]
    return events

def analyze_midi(path):
    f = open(path, 'rb')
    midi = f.read()
    f.close()
    
    midi = midi.split(b'MTrk')
    info = midi[0][8:]
    form = strb_to_int(info[:2])
    track = strb_to_int(info[2:4])
    tick = strb_to_int(info[4:])
    #print(form, track, tick)
    
    midi = midi[1:]
    for i in range(len(midi)): midi[i] = midi[i][4:]
    
    #l = analyze_track(midi[1])
    #for i in l: print(i)
    ans = [(form, track, tick)]
    for i in midi:
        ans.append(analyze_track(i, analyze_events=True))
    #    for j in l: print(j)
    #    print()
    return ans
    
def convert_insts(track, category=0):
    category = uint8_to_byte(category)
        
    for i in range(1, len(track)-1):
        ii = hex(track[i])
        ij = track[i+1]
        ih = track[i-1]
        #if ii[2] == 'c' and len(ii) == 4: print(i, ih, ii, ij)
    
    inst_change = []
    length = len(track)
    track_orig = track
    while len(track) > 1:
        dtime, count, track = parse_delta_time(track)
        cat, sub, count, event, track = parse_event(track)
        cur = (dtime, cat, hex(sub)[2:], event)
        if cur[2][0] == 'c': 
            inst_change.append(length - len(track) - 2)
            #print(length - len(track) - 2, cur)
    #print(inst_change)
    for i in inst_change:
        #print(track_orig[i-1], track_orig[i], track_orig[i+1])
        #print(bytes(category))
        track_orig = track_orig[:i+1] + bytes(category) + track_orig[i+2:]
        #track_orig[i+1] = category
        #print(track_orig[i-1], track_orig[i], track_orig[i+1])
    
    if len(inst_change) == 0:
        #print('kang')
        channels = []
        track = track_orig
        while len(track) > 1:
            dtime, count, track = parse_delta_time(track)
            cat, sub, count, event, track = parse_event(track)
            cur = (dtime, cat, hex(sub)[2:], event)
            if cur[2][0] == '8' or cur[2][0] == '9' or cur[2][0] == 'a': 
                if not int(cur[2][1], base=16) in channels: channels.append(int(cur[2][1], base=16))
        #print(channels)
        for i in channels:
            track_orig = b'\x00' + uint8_to_byte(192 + i) + category + track_orig
            #print(track_orig[0], track_orig[1], track_orig[2])
    
    return track_orig
    
def convert_midi_insts_to_piano(path, save_path, category=0):
    f = open(path, 'rb')
    midi_orig = f.read()
    f.close()
    
    midi = midi_orig.split(b'MTrk')
    info_orig = midi_orig[:14]
    info = midi[0][8:]
    form = strb_to_int(info[:2])
    track = strb_to_int(info[2:4])
    tick = strb_to_int(info[4:])
    #print(form, track, tick)
    
    ans = info_orig
    midi = midi[1:]
    heads = []
    for i in range(len(midi)): 
        heads.append(b'MTrk' + midi[i][:4])
        midi[i] = midi[i][4:]
    #print(heads)

    for i in range(len(midi)):
        conv = convert_insts(midi[i], category=category)
        len_orig = len(midi[i])
        len_conv = len(conv)
        
        #print(len_orig, len_conv)
        if len_conv != len_orig: 
            barray = []
            while len_conv > 256:
                barray.append(len_conv % 256)
                len_conv = math.floor(len_conv / 256)
            if len_conv != 0: barray.append(len_conv)
            while len(barray) < 4: barray.append(0)
            barray.reverse()
            #print(barray)
            heads[i] = heads[i][:4] + bytes(barray)

        ans = ans + heads[i]
        #print(len(midi[i]))
        #print(len(convert_insts(midi[i], category=category)))
        ans = ans + conv
    #    for j in l: print(j)
    #    print()
    #print(len(midi_orig), len(ans))
    #return ans
    f = open(save_path, 'wb')
    f.write(ans)
    f.close()
    
def track_to_heatmap(track, maxlength=240):
    stop_tick = track[-1][0]# + 1
    # map count: channel count * instrument count
    channel_count = 0
    inst_count = 0
    channel_map = {}
    inst_map = {}
    i_cur = 0
    for i in track:
        if i[1] != 'midi': continue
        c_cur = i[3]['channel']
        if i[2] == 12: i_cur = i[3]['instrument']
        if not c_cur in channel_map:
            channel_map[c_cur] = channel_count
            channel_count += 1
        if not i_cur in inst_map:
            inst_map[i_cur] = inst_count
            inst_count += 1
    channel_wise_inst_map = []
    for i in range(channel_count): channel_wise_inst_map.append([])
    for i in track:
        if i[1] != 'midi': continue
        if i[2] != 12: continue
        #print(channel_wise_inst_map, inst_map, i[3]['channel'])
        channel_wise_inst_map[channel_map[i[3]['channel']]].append(inst_map[i[3]['instrument']])
    map_count = 0
    for i in channel_wise_inst_map: map_count += len(i)
    map_map = {}
    for i in range(len(channel_wise_inst_map)): 
        if len(channel_wise_inst_map[i]) == 0: channel_wise_inst_map[i] = [0]
        for j in channel_wise_inst_map[i]:
            #print(i)
            map_map[(i, j)] = np.zeros((128, stop_tick), dtype='int16')
         
    #print(channel_map, inst_map, map_map)
    i_cur = np.zeros(12, dtype='uint8')
    for i in track:
        if i[1] != 'midi': continue
        c_cur = channel_map[i[3]['channel']]
        if i[2] == 12: 
            #print(i)
            i_cur[c_cur] = inst_map[i[3]['instrument']]
            #print(i_cur[c_cur])
        if i[2] == 8 or i[2] == 9 or i[2] == 10:
            if i[2] == 8: i[3]['strength'] = 0
            note = i[3]['note'][0] * 12 + i[3]['note'][1]
            map_map[(c_cur, i_cur[c_cur])][note, i[0]: stop_tick] = i[3]['strength']

    ans_map = {}
    for i in channel_map: 
        for j in inst_map:
            if (channel_map[i], inst_map[j]) in map_map:
                #print(i, j)
                ans_map[(i, j)] = map_map[(channel_map[i], inst_map[j])]
    
    return ans_map

def get_base_info(midi):
    ans = {'crotchet_tick': midi[0][2], 'channel_count': midi[0][1], 'structure': midi[0][0]}
    for t in midi[1:]:
        for e in t:
            if e[1] != 'meta': continue
            if e[2] != 5: continue
            #print(e)
            if len(e[3]) == 5: ans['smpte'] = e[3]
            elif len(e[3]) == 1: 
                ans['tick_dtime'] = e[3]['dtime'] / midi[0][2]
                ans['tick_per_sec'] = 1000000 / ans['tick_dtime']
            elif len(e[3]) == 3: ans['metronome_tick'] = e[3]['tick']
    return ans

def get_concatenated_heatmap(heats, info, smpl_per_tick=100):
    heatmaps = []
    for heat in heats:
        if len(heat) == 0: continue
        m = list(heat.items())[0][1]#[:, 3500:3750]
        n = np.zeros(m.shape)
        if m.shape == (128, 0): continue
        for i in heat:
            m = heat[i]#[:, 3500:3750]
            n += m
            #plt.imshow(m)
            #plt.title('channel: %d instrument: %d'%(i[0], i[1]))
            #plt.show()
        heatmaps.append(n)
        #plt.imshow(n)
        #plt.show()
        
    max_length = 0
    for i in heatmaps: max_length = max(max_length, i.shape[1])
    ans = np.zeros((128, max_length), dtype='float32')
    for i in heatmaps: 
        temp = np.zeros((128, max_length - i.shape[1]))
        ans += np.concatenate((i, temp), axis=1)
    #plt.imshow(ans)
    #plt.show()
    
    s = int(ans.shape[1] * smpl_per_tick / info['tick_per_sec'])
    ans = cv2.resize(ans, (s, 128))
    #plt.imshow(ans)
    #plt.show()
    return ans

def get_midi_map(path, smpls=100):
    midi = analyze_midi(path)
    info = get_base_info(midi)
    heats = []
    for i in midi[1:]: heats.append(track_to_heatmap(i))
    hmap = get_concatenated_heatmap(heats, info, smpl_per_tick=smpls)
    return hmap

if __name__ == '__main__':
    midi = analyze_midi('data/midi_test/1.mid')
    midi_map = get_midi_map('data/midi_test/1.mid', smpls=100)
    plt.imshow(midi_map[:, :500])
    plt.show()
    #convert_midi_insts_to_piano('203.mid', '204.mid', category=5)

