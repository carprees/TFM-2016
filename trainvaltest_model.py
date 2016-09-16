# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:15:03 2016

@author: Carprees
"""

import theano
import theano.tensor as T
import numpy

import matplotlib.pyplot as plt

#from MLP.MLP import deepMLP
#import Utilidades.RNA_util as rna
#import load.loadData as load

def train_model(x, y, index, isTrain, isBNTrain, dataset=[], batch_size=100,
                learning_rate = 0.1, L1_reg=0.00, L2_reg=0.0001, n_epochs=100,
                val=True, trainShared = True, classifier=None, allparams=[], classifier_valtest=None, allparams_valtest=[]):

    if(val):
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        test_set_x, test_set_y = dataset[2]
        # compute number of minibatches for training, validation and testing    
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size 

    else:
        train_set_x, train_set_y = dataset[0]
        test_set_x, test_set_y = dataset[1]
        # compute number of minibatches for training, validation and testing
        n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size 

    if(trainShared):
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    else:
        n_train_batches = len(train_set_x) // batch_size
#        ynew = T.cast(y, 'int32')
        
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically    
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
#    y1 = T.imatrix('y1')
#    cost = (
#        classifier.ova_svm_cost(y1)
##        + L1_reg * classifier.L1
##        + L2_reg * classifier.L2_sqr
#    )

    gparams = [T.grad(cost, param) for param in allparams]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(allparams, gparams)
    ]
    
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    if(trainShared):
        train_model_fn = theano.function(
            inputs=[index, isTrain, isBNTrain],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]
            }, on_unused_input='ignore'
        )
        
#        train_model_fn2 = theano.function(
#            inputs=[index, isTrain],
#            outputs=classifier.entropy(y),
#            givens={
#                x: train_set_x[index * batch_size: (index + 1) * batch_size],
#                y: train_set_y[index * batch_size: (index + 1) * batch_size]
#            }, on_unused_input='warn'
#        )
    else:
        train_model_fn = theano.function(
            inputs=[x, y, isTrain, isBNTrain],
            outputs=cost,
            updates=updates,
            on_unused_input='ignore'
        )
        
#        train_model_fn2 = theano.function(
#            inputs=[x, y, isTrain],
#            outputs=classifier.entropy(y),
#            on_unused_input='warn'
#        )
        
    
    test_model = theano.function(
        inputs=[index, isTrain, isBNTrain],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }, on_unused_input='ignore'
    ) 
    
#    test_model = theano.function(
#        inputs=[index, isTrain],
#        outputs=classifier.errors(y),
#        givens={
#            x: test_set_x[index * 10:(index + 1) * 10],
#            y: test_set_y[index]
#        }, on_unused_input='warn'
#    ) 

    if(val):
        validate_model = theano.function(
            inputs=[index, isTrain, isBNTrain],
            outputs=[classifier.errors(y), classifier.entropy(y)],
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }, on_unused_input='ignore'
        )
        
        best_validation_loss = numpy.inf
        val_entropy = []
        epch_val_err = 1
        
    best_params = allparams
    train_cost = []
#    train_entr = []
    ep = []
    
    for epoch in range(1, n_epochs+1):
        ep.append(epoch)
        if(trainShared):
            minibatch_avg_cost = [train_model_fn(i, isTrain=True, isBNTrain=True) for i in range(n_train_batches)]
        else:
            minibatch_avg_cost = [train_model_fn(train_set_x[i * batch_size:(i + 1) * batch_size], 
                                                 train_set_y[i * batch_size:(i + 1) * batch_size].astype(numpy.int32), 
                                                 isTrain=True, isBNTrain=True) for i in range(n_train_batches)]
        
        train_cost.append(numpy.mean(minibatch_avg_cost))
        
#        train_entropy = [train_model_fn2(i, False) for i in range(n_train_batches)]
#        
#        train_entr.append(numpy.mean(train_entropy))
        
        test_losses = [test_model(i, False, True) for i in range(n_test_batches)]
#        test_losses = [test_model(i, False) for i in range(7178)]
        test_score = numpy.mean(test_losses)
        
        # Only true if we have validation data
        if(val):
            validation_losses = [validate_model(i, False, True) for i
                                 in range(n_valid_batches)]
                                 
                                 
            this_validation_loss = numpy.mean([a[0] for a in validation_losses])
            
            val_entropy.append(numpy.mean([a[1] for a in validation_losses]))
            
            print('epoch %i, validation error %f %%, test error %f %%' % 
                  (epoch, this_validation_loss * 100., test_score * 100.))
            
            if this_validation_loss < best_validation_loss:
                best_validation_loss = this_validation_loss
                best_params = allparams
                epch_val_err = epoch
            
        else:
#            print('epoch %i, train cost %f %%, test error %f %%' % (epoch, train_entropy[epoch-1], test_score * 100.))
            print('epoch %i, train cost %f %%, test error %f %%' % (epoch, train_cost[epoch-1], test_score * 100.))
            best_params=allparams
    
#    plt.axis([1, n_epochs, -1, 3])    
    # red dashes, blue squares and green triangles
    if(val):
        plt.plot(ep, train_cost, 'b', ep, val_entropy, 'r', (epch_val_err, epch_val_err), (0, 3), 'k-')
    else:
        plt.plot(ep, train_cost, 'b')
    plt.show()   
    
    if(val):
        print('Best validation error obtained at epoch %i' % (epch_val_err))
        print('Best validation entropy %f, obtained at epoch %i, with a train cost of %f' 
            % (min(val_entropy), val_entropy.index(min(val_entropy))+1, train_cost[val_entropy.index(min(val_entropy))]))
      
    return best_params      
        
def testall_model(index, x, y, isTrain, batch_size=100, classifier=None, allparams=[], dataset=[], val = True):   
    
    test_set_x, test_set_y = dataset[-1]
    
    if(val):
        train_set_x, train_set_y = dataset[0]
        valid_set_x, valid_set_y = dataset[1]
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
        
        validate_model = theano.function(
            inputs=[index, isTrain],
            outputs=classifier.errors(y),
            givens={
                x: valid_set_x[index * batch_size:(index + 1) * batch_size],
                y: valid_set_y[index * batch_size:(index + 1) * batch_size]
            }, on_unused_input='warn'
        )

    else:
        train_set_x, train_set_y = dataset[0]
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size       
    
    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index, isTrain],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }, on_unused_input='warn'
    ) 
    
    test_losses = [test_model(i, False) for i in range(n_test_batches)]
    test_score = numpy.mean(test_losses)
    
    if(val):
        validation_losses = [validate_model(i, False) for i in range(n_valid_batches)]
        valid_score = numpy.mean(validation_losses)
    
        print(('Optimization complete. Best validation performance %f %%. Test performance %f %%') %
            (valid_score * 100., test_score * 100.))    
    else:
        print(('Optimization complete. Test performance %f %%') %
            (test_score * 100.)) 
            
            
            
            
        
    