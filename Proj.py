# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:41:44 2016

@author: Carprees
"""


import Utilidades.RNA_util as rna  
from test import start_test, continue_test, special_test    
    
#from testCNN import test_CNN     
#test_CNN(
#        learning_rate=0.1, 
#        L1_reg=0.00, 
#        L2_reg=0.0001, 
#        n_epochs=20,
#        dataset='emotions2.pkl.gz', 
#        batch_size=500, 
#        n_hidden=[512, 512], 
#        n_out=7, 
#        activation=rna.relu(),
#        p_drop = 0.5,
#        val=True,
#        im_dim=[48, 48], 
#        n_kerns=[10,10,10],
#        kerns_shape = [[3,3],[3,3],[3,3]],
#        poolsize = (2, 2)
#        )
    
### Para entrenar desde 0 ###
     
#start_test(
#        learning_rate=0.1, 
#        L1_reg=0.00, 
#        L2_reg=0.0001, 
#        n_epochs=30,
#        dataset='data/new_emotions_datAug3.h5', 
#        batch_size=194, 
#        n_hidden=[1024], 
#        n_out=7, 
#        activation=rna.relu(),
#        batch_norm = True,
#        p_drop = 0.5,
#        p_dropConv = [0.2, 0.3, 0.4, 0.5],
#        gaussian_std=0.,
#        val=False,
#        trainShared=False,
#        im_dim=[42, 42], 
#        savename='TrainedModels/mierda.pkl'
#        )
        
### Para seguir entrenando un modelo anterior ###  
  
#continue_test(
#        learning_rate=0.01, 
#        L1_reg=0.00, 
#        L2_reg=0.0001, 
#        n_epochs=10,
#        dataset='data/new_emotions_datAug2.h5', 
#        model='TrainedModels/replica_OxfordRedDatAugEntr.pkl',
#        batch_size=194, 
#        n_hidden=[1024], 
#        n_out=7, 
#        activation=rna.relu(),
#        batch_norm = True,
#        p_drop = 0.5,
#        p_dropConv = [0.2, 0.3, 0.4, 0.5],
#        gaussian_std=0.,
#        val=False,
#        trainShared=False,
#        im_dim=[42, 42], 
#        savename='TrainedModels/replica_OxfordRedDatAugEntr_cont.pkl'
#        )  
 
### Para testear un modelo anterior ### 
       
special_test(
        dataset='data/new_emotions_datAug.h5', 
        model='TrainedModels/mierda.pkl',
        batch_size=3589, # 194, 3589, 1794, 2392, 1435, 1196 # train 3022
        n_hidden=[1024], 
        n_out=7, 
        activation=rna.relu(),
        batch_norm=True,
        p_drop = 0.5,
        p_dropConv = [0.2, 0.3, 0.4, 0.5],
        gaussian_std=0.,
        im_dim=[42, 42], 
        )



