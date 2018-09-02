# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 18:57:25 2018

@author: BillStark001
"""

import os
from pyquery import PyQuery as pq
from tqdm import tqdm
import urllib

source_list = 'midi_.html'
save_dir = 'midis/'

def get_source_dict(html=source_list):
    f = open(html, 'r', encoding='utf-8')
    s = f.readlines()
    d = {}
    for i in s: 
        ii = i[:-1]
        ii = i.split('</a>')[0].split('<a href=\"')
        if len(ii) <= 1: continue
        ii = ii[1].split('\">')
        #print(ii)
        d[ii[1]] = ii[0]
    return d

def get_html(url):
    html = urllib.request.urlopen(url).readlines()
    for i in range(len(html)):
        html[i] = html[i].decode(encoding='utf-8')
    return html

def get_source_htmls():
    sources = get_source_dict()
    sources_ = {}
    for d in tqdm(sources):
        print(d)
        sources_[d] = []
        htmls = get_html(sources[d])
        s = ''
        for i in htmls:
            if i.startswith('		<section class="article-content">') or i.startswith('				目前共收录'):
                s = i
                break
        s = s.split('<a ')
        for i in s:
            if i.startswith('href=\"'):
                i = i.split('</a>')[0][6:].split('\">')
                if i[0].endswith('.zip') or i[0].endswith('.mid'): 
                    i[1] = i[1].strip(' ')
                    sources_[d].append(i)
    return sources_
                
def download(url, d, name):
    #print(url)
    try:
        os.mkdir(d)
    except:
        pass
    #print(d)
    print(name)
    def Schedule(a,b,c):
        '''''
        a:已经下载的数据块
        b:数据块的大小
        c:远程文件的大小
        '''
        per = 100.0 * a * b / c
        if per > 100 :
            per = 100
        print('%.2f%%' % per)
    local = os.path.join(d, name)
    try:
        urllib.request.urlretrieve(url,local)
    except:
        pass

try:
    assert sources
except:
    sources = get_source_htmls()
    
if __name__ == '__main__':
    for cat in sources:
        for i in tqdm(sources[cat]):
            url = i[0]
            d = save_dir + cat + '/'
            name = i[1] + '.' + i[0].split('.')[-1]
            download(url, d, name)