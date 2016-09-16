# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 11:07:38 2016

@author: Carprees
"""
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import cv2
import random as rand

def dropout(input, isTrain, p=0): 
    if p > 0:
        srng = RandomStreams(rand.randint(1, 2147462579))
        retain_prob = 1 - p
        input = theano.ifelse.ifelse(isTrain, input*srng.binomial(input.shape, p=retain_prob, dtype=theano.config.floatX), 
                                     input*retain_prob)
    return input
                                     
def dropout_conv(input, isTrain, p=0, n_filters = 1, im_shape = 2): 
    if p > 0:
        srng = RandomStreams(rand.randint(1, 2147462579))
        retain_prob = 1 - p
        input = theano.ifelse.ifelse(isTrain, input*srng.binomial((n_filters,), p=retain_prob, dtype=theano.config.floatX).dimshuffle('x', 0,'x','x'), 
                                     input*retain_prob)
    return input

def batch_norm(input, gamma, beta, mean, std):
    return T.nnet.bn.batch_normalization(input, gamma, beta, mean=mean, std=std, mode='high_mem')
                                                     
def logReg_batch_norm(input, gamma, beta):
    return theano.tensor.nnet.bn.batch_normalization(input, gamma, beta, 
                                                     input.mean((0,), keepdims=True), 
                                                     std =  T.ones_like(input.var((0,), keepdims = True)),
                                                     mode='high_mem')

def online_dat_aug(data, shift = 4, rot = 45):   
    arr=[]

    smed=shift/2
    M = cv2.getRotationMatrix2D((48/2,48/2),rot,1)
    M2 = cv2.getRotationMatrix2D((48/2,48/2),-rot,1)
    red = float((48-shift))/float(48)    
    resi = (58-48)/2+smed    
    
    np.random.randint(7, size=10)
    
    for i in range(len(data)):
        img = np.reshape(np.ravel(data[i]), (48, 48))
        
        if(np.random.randint(2, size=1)==1):
            img = cv2.flip(img, 1)
         
        ran = np.random.randint(7, size=1)

        if(ran==1):
            # 3 píxel a derecha
            crop=img[shift/2:48-shift/2,shift:48]
            arr.append(np.ravel(crop))
            
        elif(ran==2):
            # 3 píxel a izquierda
            crop=img[shift/2:48-shift/2,0:48-shift]
            arr.append(np.ravel(crop))

        elif(ran==3):
            # rotación 45 grados
            dst = cv2.warpAffine(img,M,(48,48))
            crop=dst[smed:48-smed,smed:48-smed]
            arr.append(np.ravel(crop))
    
        elif(ran==4):
            # rotacion -45 grados
            dst = cv2.warpAffine(img,M2,(48,48))
            crop=dst[smed:48-smed,smed:48-smed]
            arr.append(np.ravel(crop))
        
        elif(ran==5):
            # escala 1.2
            res = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
            crop=res[resi:len(res)-resi,resi:len(res)-resi]
            arr.append(np.ravel(crop))

        elif(ran==6):
            # escala 0.88
            res = cv2.resize(img,None,fx=red, fy=red, interpolation = cv2.INTER_CUBIC)
            crop=res
            arr.append(np.ravel(crop))

        else:            
            crop=img[smed:48-smed,smed:48-smed]
            arr.append(np.ravel(crop))
            
    final = np.array(arr, dtype=np.float32)        
    return final
    
def data_various_aug(train, shift = 4):
    data=train 
    arr=[]
    
    smed=shift/2
    M = cv2.getRotationMatrix2D((48/2,48/2),45,1)
    M2 = cv2.getRotationMatrix2D((48/2,48/2),-45,1)
    
    resi = (58-48)/2+smed    
    
    for i in range(0,len(data)): 

        dat = np.reshape(np.ravel(data[i]), (48, 48))
               
        crop = np.zeros((48-shift, 48-shift))
        
        # Crop central
        crop=dat[smed:48-smed,smed:48-smed]
        arr.append(np.ravel(crop))
        
        # 3 pixel a derecha
        crop=dat[shift/2:48-shift/2,shift:48]
        arr.append(np.ravel(crop))
        
        # 3 píxel a izquierda
        crop=dat[shift/2:48-shift/2,0:48-shift]
        arr.append(np.ravel(crop))
        
        # rotación 45 grados
        dst = cv2.warpAffine(dat,M,(48,48))
        crop=dst[smed:48-smed,smed:48-smed]
        arr.append(np.ravel(crop))
        
        # rotacion -45 grados
        dst = cv2.warpAffine(dat,M2,(48,48))
        crop=dst[smed:48-smed,smed:48-smed]
        arr.append(np.ravel(crop))
        
        # escala 1.2
        res = cv2.resize(dat,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
        crop=res[resi:len(res)-resi,resi:len(res)-resi]
        arr.append(np.ravel(crop))
        
        # escala 0.88
        res = cv2.resize(dat,None,fx=0.88, fy=0.88, interpolation = cv2.INTER_CUBIC)
        crop=res
        arr.append(np.ravel(crop))

        
    finaldata=np.array(arr, dtype=np.float32)    
    
    return finaldata

def mirror_dataset(data):  
    newdata = []
    
    for i in range(len(data)):
        newdata.append(data[i]) 
        newdata.append(np.ravel(cv2.flip(np.reshape(np.ravel(data[i]), (48, 48)),1)))
    
    finaldata = np.array(newdata, dtype=np.float32)  
    return finaldata
    
def center_crop(data, shift = 4):
    
    smed=shift/2
    arr=[]
    
    for i in range(0,len(data)): 
        dat = np.reshape(np.ravel(data[i]), (48, 48))        
        
        crop=dat[smed:48-smed,smed:48-smed]
        arr.append(np.ravel(crop))
        
    finaldata=np.array(arr, dtype=np.float32) 
    
    return finaldata
                                              
def spatial_2d_padding(x, padding=(1, 1), dim_ordering='th'):
    '''Pad the 2nd and 3rd dimensions of a 4D tensor
    with "padding[0]" and "padding[1]" (resp.) zeros left and right.
    '''
    input_shape = x.shape
    if dim_ordering == 'th':
        output_shape = (input_shape[0],
                        input_shape[1],
                        input_shape[2] + 2 * padding[0],
                        input_shape[3] + 2 * padding[1])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(None),
                   slice(padding[0], input_shape[2] + padding[0]),
                   slice(padding[1], input_shape[3] + padding[1]))

    elif dim_ordering == 'tf':
        output_shape = (input_shape[0],
                        input_shape[1] + 2 * padding[0],
                        input_shape[2] + 2 * padding[1],
                        input_shape[3])
        output = T.zeros(output_shape)
        indices = (slice(None),
                   slice(padding[0], input_shape[1] + padding[0]),
                   slice(padding[1], input_shape[2] + padding[1]),
                   slice(None))
    else:
        raise Exception('Invalid dim_ordering: ' + dim_ordering)
    return T.set_subtensor(output[indices], x)
    
def gaussian_perturb(input, std=0., rng=None):
    """Return a Theano variable which is perturbed by additive zero-centred
    Gaussian noise with standard deviation ``std``.
    Parameters
    ----------
    arr : Theano variable
        Array of some shape ``n``.
    std : float or scalar Theano variable
        Standard deviation of the Gaussian noise.
    rng : Theano random number generator, optional [default: None]
        Generator to draw random numbers from. If None, rng will be
        instantiated on the spot
    """
    if std > 0.0:
        if rng is None:
            rng = RandomStreams()
        noise = rng.normal(size=input.shape, std=std)
        noise = T.cast(noise, theano.config.floatX)
        return input + noise
    else:
        return input

def valFromTrain():
    import gzip, six.moves.cPickle as pickle
    
    with gzip.open('data/emotions.pkl.gz', 'rb') as f:
        train_set,test_set = pickle.load(f)
    
        data = train_set[0]
        label = train_set[1]   
        
        np.random.seed(100)
        np.random.shuffle(data)
        
        np.random.seed(100)
        np.random.shuffle(label)
        
        n_valid = int(round(0.2 * len(data)))                
                       
        valid_data = data[len(data)-n_valid:len(data)]
        
        valid_label = label[len(label)-n_valid:len(label)]
        
        valid_set = (valid_data, valid_label)
        
        train_set = (data[0:len(data)-n_valid],label[0:len(label)-n_valid]) 
        
        print (train_set, valid_set)
    
def relu():
    return T.nnet.relu
    
def tanh():
    return T.tanh
    
def sigmoid():
    return T.nnet.sigmoid