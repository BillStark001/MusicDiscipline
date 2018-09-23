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
from keras.layers import Dense, LSTM, Dropout, Activation, Input
from keras.optimizers import Adam

seq_length = 25
data_dims_g = (seq_length, 512)
data_dims_d = (seq_length, 128)

def generator():
    seq_in = Input(shape=data_dims_g, name='in__')
    x = LSTM(256, return_sequences=True, name='lstm_256')(seq_in)
    x = Dropout(0.5)(x)
    seq_out = LSTM(128, return_sequences=True, name='out__')(x)
    model = Model(seq_in, seq_out)
    return model

def discriminator():
    seq_in = Input(shape=data_dims_d, name='in__')
    x = LSTM(64, return_sequences=True, name='lstm_64')(seq_in)
    x = Dropout(0.5)(x)
    x = LSTM(32, return_sequences=False, name='lstm_32')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, name='out__')(x)
    model = Model(seq_in, out)
    return model

#discriminator()(generator().output)

class LSGAN():
    
    def __init__(self, data_loader, optimizer=Adam(0.0002, 0.5)):
        
        self.data_loader = data_loader
        optimizer = optimizer

        #Build Generator
        self.generator = generator()
        #self.generator.compile(loss='mse',
        #    optimizer=optimizer,
        #    metrics=['accuracy'])
        
        #Build Discriminator
        self.discriminator = discriminator()
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.discriminator.trainable = False
        
        #Combine
        seq_in = Input(shape=data_dims_g)
        gen = self.generator(seq_in)
        valid = self.discriminator(gen)
        self.combined = Model(seq_in, valid)
        self.combined.compile(loss='mse', optimizer=optimizer)

    def train(self, epochs, batch_size=32, seq_length=25):
        
        #Load Data
        data_loader = self.data_loader(length=seq_length, batch_size=batch_size)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            
            X, Y = next(data_loader)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            
            Y_gen = self.generator.predict(X)
            
            d_loss_real = self.discriminator.train_on_batch(Y, valid)
            d_loss_fake = self.discriminator.train_on_batch(Y_gen, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            
            # ---------------------
            #  Train Generator
            # ---------------------
            
            g_loss = self.combined.train_on_batch(X, valid)
            
            # Plot the progress
            print ("Epoch %d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    
    def save_weights(self, path):
        self.combined.save_weights(path)
        
    def load_weights(self, path):
        self.combined.load_weights(path)
        
    def predict(self, seq):
        return self.generator.predict(seq)
    '''
            # If at save interval => save generated image samples
            #if epoch % sample_interval == 0:
            #    self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()
    '''