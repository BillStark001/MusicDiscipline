# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:08:33 2018

@author: Zhao
"""

import data_loader
import models
import seq_models

import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler

def lr_schedule(epoch):
    lr_ans = 0.0005
    lr = 10
    if epoch > 240:
        lr **= -3
    elif epoch > 200:
        lr **= -2.5
    elif epoch > 160:
        lr **= -2
    elif epoch > 120:
        lr **= -1.5
    elif epoch > 80:
        lr **= -1
    elif epoch > 40:
        lr **= -0.5
    else:
        lr **= 0
    return lr * lr_ans

weights_path = 'weights.h5'

#model = models.LSGAN(data_loader=data_loader.gen_data)#, optimizer=SGD())
model = seq_models.AttentionSeq2Seq(input_dim=128, input_length=50, hidden_dim=128, output_length=50, output_dim=128, depth=1)
model.compile(loss='mse', optimizer='Adam')

if __name__ == '__main__':
    
    try:
        #model.train_gen(25)
        model.fit_generator(data_loader.gen_fake_data(50, 48), steps_per_epoch=48, epochs=250, 
                            callbacks=[LearningRateScheduler(lr_schedule)])
    except KeyboardInterrupt as e:
        pass
    model.save_weights(weights_path)
    
    #model.load_weights(weights_path)
    
    gen = data_loader.gen_fake_data(length=750, batch_size=1)
    x, y = next(gen)
    y_ = y[:, :0, :]
    for i in range(20):
        y_ = np.concatenate((y_, model.predict(x[:, i*25+12: i*25+62])[:, 12:37]), axis=1)
    plt.imshow(y_[0, :100])
    plt.show()
    plt.imshow(y[0, 12:112])
    plt.show()