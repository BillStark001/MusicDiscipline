# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:08:33 2018

@author: Zhao
"""

import data_loader
import models

weights_path = 'weights.h5'

model = models.LSGAN(data_loader=data_loader.gen_data)

if __name__ == '__main__':
    model.train(2000)
    model.save_weights(weights_path)