# -*- coding: utf-8 -*-
"""
Created on Sun Sep  2 16:22:02 2018
STILL UNDER CONSTRUCTION! STILL UNDER CONSTRUCTION! STILL UNDER CONSTRUCTION!
@author: BillStark001
"""

import keras
from keras import backend as K
import numpy as np

from keras.models import Model
from keras.layers import Dense, LSTM, Dropout

seq_length = 25
data_dims_g = (seq_length, 256, 2)
data_dims_d = (seq_length, 128)

def generator():
    model = Model()
    x = LSTM(200, input_shape=data_dims_g)
    x = LSTM(200)(x)