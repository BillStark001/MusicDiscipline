# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 22:08:33 2018

@author: Zhao
"""

import data_loader
import models

model = models.LSGAN(data_loader=data_loader.gen_data)
model.train(10000)