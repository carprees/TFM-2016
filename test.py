# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 15:24:41 2016

@author: Carprees
"""

from __future__ import print_function

import os, sys
import theano
import theano.tensor as T
import numpy
import timeit
import six.moves.cPickle as pickle

from MLP.MLP import deepMLP
import CNN.CNN as cnn
import Utilidades.RNA_util as rna
import Utilidades.Util as ut
import load.loadData as load
import trainvaltest_model as m

def start_test(learning_rate=0.1, L1_reg=0.00, L2_reg=0.0001, n_epochs=200, dataset='mnist.pkl.gz', 
             batch_size=500, n_hidden=[1024], n_out=10, activation=rna.tanh(), batch_norm = False, p_drop = 0.0, 
             p_dropConv=[0], gaussian_std=0., val=False, trainShared=False, im_dim=[28, 28], savename='TrainedModels/model.pkl'):
                 
    datasets = load.load_data(dataset, val, trainShared=trainShared)
    
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    isTrain = T.bscalar('isTrain') 
    isBNTrain = T.bscalar('isBNTrain')
    
    rng = numpy.random.RandomState(1234)

    #### Parte convolucional ####
    # OxfordNet11LayerRed, OxfordNet11Layer, YichuanConv
    # ResidualNet18Layer, ResidualNetOxfordMod, MiniResidual, ResidualNet18LayerRed

    conv = cnn.OxfordNet11LayerRed (
        rng,
        input=x,
        isTrain=isTrain,
        isBNTrain=isBNTrain,
        batch_size=batch_size,
        im_dim=im_dim,
        activation=activation,
        batch_norm = batch_norm,
        p_drop=p_dropConv,
        gaussian_std=gaussian_std
    )
    
    for i in range(len(conv.convLayers)):
        print(conv.convLayers[i].im_dim_out)

    #### Parte fully connected ####
    classifier = deepMLP(
        rng=rng,
        input=conv.output.flatten(2),
        isTrain=isTrain,
        isBNTrain=isBNTrain,
        n_in=conv.n_out,
        n_hidden=n_hidden,
        n_out=n_out,
        activation=activation,
        batch_norm = batch_norm,
        p_drop = p_drop,
        gaussian_std=gaussian_std
    )
    
    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    allparams = classifier.params + conv.params
   
    ###############
    # TRAIN MODEL #
    ###############
    start_time = timeit.default_timer()
    print('... training')

    best_params = m.train_model(index=index, x=x, y=y, isTrain=isTrain, isBNTrain=isBNTrain, dataset=datasets, batch_size=batch_size, 
                  learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg, n_epochs=n_epochs, val=val, trainShared=trainShared,
                  classifier=classifier, allparams=allparams)    
  
    f = open(savename, 'wb')
    pickle.dump(best_params, f)
    
    f.close()

    end_time = timeit.default_timer()
  
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)

def continue_test(learning_rate=0.1, L1_reg=0.00, L2_reg=0.0001, n_epochs=200, dataset='mnist.pkl.gz',
             model='TrainedModels/model.pkl', batch_size=500, n_hidden=[1024], n_out=10, activation=rna.tanh(), batch_norm = False,
             p_drop = 0.0, p_dropConv=[0], gaussian_std=0., val=True, trainShared=False, im_dim=[28, 28], savename='TrainedModels/model2.pkl'):
                 
    datasets = load.load_data(dataset, val, trainShared=trainShared)
    
    set_x, set_y = datasets[1]

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    isTrain = T.bscalar('isTrain')
    isBNTrain = T.bscalar('isBNTrain')
    
    rng = numpy.random.RandomState(1234)

    #### Parte convolucional ####
    conv = cnn.OxfordNet11LayerRed (
        rng,
        input=x,
        isTrain=isTrain,
        isBNTrain=isBNTrain,
        batch_size=batch_size,
        im_dim=im_dim,
        activation=activation,
        batch_norm = batch_norm,
        p_drop=p_dropConv,
        gaussian_std=gaussian_std
    )
    
    for i in range(len(conv.convLayers)):
        print(conv.convLayers[i].im_dim_out)

    #### Parte fully connected ####
    classifier = deepMLP(
        rng=rng,
        input=conv.output.flatten(2),
        isTrain=isTrain,
        isBNTrain=isBNTrain,
        n_in=conv.n_out,
        n_hidden=n_hidden,
        n_out=n_out,
        activation=activation,
        batch_norm = batch_norm,
        p_drop = p_drop,
        gaussian_std=gaussian_std
    )

    ### Cargamos parámetros de W y bias para seguir entrenando un modelo anterior ###
    fi = open(model, 'rb')
    lastparams=pickle.load(fi)
    
    fi.close()
    
    allparams = classifier.params + conv.params

    updates = [
        (param, modparam)
        for param, modparam in zip(allparams, lastparams)
    ]
    
    modify_model = theano.function(
        inputs=[index],
        updates=updates,
        outputs=2*index,
        on_unused_input='ignore'
    )
    
    modify_model(0)
    
    ###############
    # TRAIN MODEL #
    ###############
    start_time = timeit.default_timer()
    print('... training')

    best_params = m.train_model(index=index, x=x, y=y, isTrain=isTrain, isBNTrain=isBNTrain, dataset=datasets, batch_size=batch_size, 
                  learning_rate=learning_rate, L1_reg=L1_reg, L2_reg=L2_reg, n_epochs=n_epochs, val=val, trainShared=trainShared,
                  classifier=classifier, allparams=allparams)    
  
    f = open(savename, 'wb')
    pickle.dump(best_params, f)
    
    f.close()

    end_time = timeit.default_timer()
  
    print(('The code for file ' +
           os.path.split(__file__)[1] +
           ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
           
def special_test(dataset='mnist.pkl.gz', model='TrainedModels/model.pkl',batch_size=500, n_hidden=[1024], 
                           n_out=10, activation=rna.tanh(), batch_norm=False, p_drop = 0.0, p_dropConv=[0], gaussian_std=0.,
                            im_dim=[28, 28]):

    import h5py
    with h5py.File(dataset,'r') as hf:
        data3 = hf.get('traindata')
        traindata = numpy.array(data3)
        traindata=traindata/numpy.max(traindata)
        numpy.random.seed(100)
        numpy.random.shuffle(traindata)
        traindata = traindata[0:batch_size*2]
        data = hf.get('testdataAug')
        testdata = numpy.array(data)[0:50246]#
        testdata=testdata/numpy.max(testdata)
        data1 = hf.get('testlabel')
        testlabel = numpy.array(data1)[0:3589]#
     
    print(len(testdata))  
 
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
    isTrain = T.bscalar('isTrain')
    isBNTrain = T.bscalar('isBNTrain')

    rng = numpy.random.RandomState(1234)

    #### Parte convolucional ####
    # OxfordNet11LayerRed, OxfordNet11Layer, YichuanConv
    # ResidualNet18Layer, ResidualNetOxfordMod, MiniResidual, ResidualNet18LayerRed
    conv = cnn.OxfordNet11LayerRed (
        rng,
        input=x,
        isTrain=isTrain,
        isBNTrain=isBNTrain,
        batch_size=batch_size,
        im_dim=im_dim,
        activation=activation,
        batch_norm=batch_norm,
        p_drop=p_dropConv,
        gaussian_std=gaussian_std    
    )
    
    for i in range(len(conv.convLayers)):
        print(conv.convLayers[i].im_dim_out)
    
    #### Parte fully connected ####
    classifier = deepMLP(
        rng=rng,
        input=conv.output.flatten(2),
        isTrain=isTrain,
        isBNTrain=isBNTrain,
        n_in=conv.n_out,
        n_hidden=n_hidden,
        n_out=n_out,
        activation=activation,
        batch_norm=batch_norm,
        p_drop = p_drop,
        gaussian_std=gaussian_std
    )
    
    ### Cargamos parámetros de W y bias para seguir entrenando un modelo anterior ###
    fi = open(model, 'rb')
    lastparams=pickle.load(fi)
    
    fi.close()
    
    allparams = classifier.params + conv.params

    updates = [
        (param, modparam)
        for param, modparam in zip(allparams, lastparams)
    ]
    
    modify_model = theano.function(
        inputs=[index],
        updates=updates,
        outputs=2*index,
        on_unused_input='ignore'
    )
    
    modify_model(0)
    
    ### Modificamos los parámetros mean y std de la red ###
    
    if(batch_norm):        
        allBNparams = conv.BN_params + classifier.BN_params
        
        extractMeanSTD = theano.function(
            inputs=[x, isTrain, isBNTrain],
            outputs=[conv.convLayers[0].mean, conv.convLayers[0].std,
                     conv.convLayers[1].mean, conv.convLayers[1].std,
                     conv.convLayers[2].mean, conv.convLayers[2].std,
                     conv.convLayers[3].mean, conv.convLayers[3].std,  
                     classifier.hiddenLayers[0].mean, classifier.hiddenLayers[0].std
#                     classifier.hiddenLayers[1].mean, classifier.hiddenLayers[1].std     
                     ],
            on_unused_input='ignore'
        )
        
#        extractMeanSTD = theano.function(
#            inputs=[x, isTrain, isBNTrain],
#            outputs=[conv.convLayers[0].mean, conv.convLayers[0].std,
#                 conv.convLayers[1].ResidualLayer[0].mean, conv.convLayers[1].ResidualLayer[0].std,
#                 conv.convLayers[1].ResidualLayer[1].mean, conv.convLayers[1].ResidualLayer[1].std,
#                 conv.convLayers[2].ResidualLayer[0].mean, conv.convLayers[2].ResidualLayer[0].std,
#                 conv.convLayers[2].ResidualLayer[1].mean, conv.convLayers[2].ResidualLayer[1].std,
#                 conv.convLayers[3].ResidualLayer[0].mean, conv.convLayers[3].ResidualLayer[0].std,
#                 conv.convLayers[3].ResidualLayer[1].mean, conv.convLayers[3].ResidualLayer[1].std,
#                 conv.convLayers[4].ResidualLayer[0].mean, conv.convLayers[4].ResidualLayer[0].std,
#                 conv.convLayers[4].ResidualLayer[1].mean, conv.convLayers[4].ResidualLayer[1].std,
#                 conv.convLayers[5].ResidualLayer[0].mean, conv.convLayers[5].ResidualLayer[0].std,
#                 conv.convLayers[5].ResidualLayer[1].mean, conv.convLayers[5].ResidualLayer[1].std,
#                 conv.convLayers[6].ResidualLayer[0].mean, conv.convLayers[6].ResidualLayer[0].std,
#                 conv.convLayers[6].ResidualLayer[1].mean, conv.convLayers[6].ResidualLayer[1].std,
#                 classifier.hiddenLayers[0].mean, classifier.hiddenLayers[0].std  
#                 ],
#                 on_unused_input='ignore'
#        )
#        
#        extractMeanSTD = theano.function(
#            inputs=[x, isTrain, isBNTrain],
#            outputs=[conv.convLayers[0].mean, conv.convLayers[0].std,
#                     conv.convLayers[1].mean, conv.convLayers[1].std,
#                     conv.convLayers[2].ResidualLayer[0].mean, conv.convLayers[2].ResidualLayer[0].std,
#                     conv.convLayers[2].ResidualLayer[1].mean, conv.convLayers[2].ResidualLayer[1].std,
#                     classifier.hiddenLayers[0].mean, classifier.hiddenLayers[0].std  
#                     ],
#            on_unused_input='ignore'
#        )
        
        algo = extractMeanSTD(traindata[0:batch_size], False, True)
        
#        algo2 = extractMeanSTD(traindata[3589:7178], False, True)
#        
#        algo3 = numpy.mean( numpy.array([ algo, algo2 ]), axis=0 )
    
        updatesBN = [
            (BNparam, newBNparam)
            for BNparam, newBNparam in zip(allBNparams, algo)
        ]
        
        modifyBN_model = theano.function(
            inputs=[index],
            updates=updatesBN,
            outputs=2*index,
            on_unused_input='ignore'
        )
        
        modifyBN_model(0)
    
    ###############
    # TEST MODEL #
    ###############
    print('... testing')
    
    n_test_batches = len(testdata) // batch_size 
    
    print (n_test_batches)
    
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    specialtest_model = theano.function(
        inputs=[x, isTrain, isBNTrain],
        outputs=classifier.logRegressionLayer.linear,
        on_unused_input='ignore'
    ) 
    
    logRegression_linear = [specialtest_model(testdata[i*batch_size:(i+1)*batch_size], False, False).tolist() for i in range(n_test_batches)]
    
    logRegression_linear = numpy.array(ut.unlist(logRegression_linear), dtype = numpy.float32)
        
    xnew = T.matrix()  
    
    su = T.sum(xnew, axis = 0)
    
    p_y_given_x = T.nnet.softmax(su) 
    
    y_pred = T.argmax(p_y_given_x, axis=1)
    
    test_model = theano.function(
        inputs=[xnew, y],
        outputs=T.neq(y_pred, y),
        on_unused_input='ignore'
    ) 
    
    num = len(testdata)/len(testlabel)
    
    test_losses = [test_model(logRegression_linear[i*num:(i+1)*num], testlabel[i:i+1]) for i in range(len(testlabel))]
    
    test_score = numpy.mean(test_losses)
   
    print(('Optimization complete. Test performance %f %%') % (test_score * 100.)) 