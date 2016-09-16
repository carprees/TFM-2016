# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 12:18:43 2016

@author: Carprees
"""

import os, sys
import theano
import theano.tensor as T
import numpy
import gzip
import six.moves.cPickle as pickle
import h5py

def load_data(dataset, val, trainShared = True):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    
    :type val: boolean
    :param val: flag to load validation or not
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz' or data_file == 'emotions.pkl.gz':
            dataset = new_path

    print('... loading data')
    
    spl = dataset.split('.')
    
    if(spl[1]=='h5'):  
        if(val):
            try:
                print('Entro aqui 1')
                with h5py.File(dataset,'r') as hf:
                    data = hf.get('trainvaliddata')
                    traindata = numpy.array(data)
                    data1 = hf.get('trainvalidlabel')
                    trainlabel = numpy.array(data1)
                    data2 = hf.get('testdata')
                    testdata = numpy.array(data2)
                    data3 = hf.get('testlabel')
                    testlabel = numpy.array(data3)
                    data4 = hf.get('validdata')
                    validdata = numpy.array(data4)
                    data5 = hf.get('validlabel')
                    validlabel = numpy.array(data5)
                    
                    numpy.random.seed(100)
                    numpy.random.shuffle(traindata)
                    
                    numpy.random.seed(100)
                    numpy.random.shuffle(trainlabel)
                    
                    train_set = (traindata/numpy.max(traindata), trainlabel)
                    test_set = (testdata/numpy.max(testdata), testlabel)
                    valid_set = (validdata/numpy.max(validdata), validlabel)
                
            except:
                print('Entro aqui')
                with h5py.File(dataset,'r') as hf:
                    data = hf.get('traindata')
                    traindata = numpy.array(data)
                    data1 = hf.get('trainlabel')
                    trainlabel = numpy.array(data1)
                    data2 = hf.get('testdata')
                    testdata = numpy.array(data2)
                    data3 = hf.get('testlabel')
                    testlabel = numpy.array(data3)
                
                numpy.random.seed(100)
                numpy.random.shuffle(traindata)
                
                numpy.random.seed(100)
                numpy.random.shuffle(trainlabel)
                
                train_set = (traindata/numpy.max(traindata), trainlabel)
                test_set = (testdata/numpy.max(testdata), testlabel)
                
                n_valid = int(round(0.2 * len(traindata)))                
                               
                valid_data = traindata[len(traindata)-n_valid:len(traindata)]
                
                valid_label = trainlabel[len(trainlabel)-n_valid:len(trainlabel)]
                
                valid_set = (valid_data, valid_label)
                
                train_set = (traindata[0:len(traindata)-n_valid],trainlabel[0:len(trainlabel)-n_valid])
                print(len(train_set[0]), len(valid_set[0]),len(test_set[0]))  
                
        else:
            with h5py.File(dataset,'r') as hf:
                data = hf.get('traindata')
                traindata = numpy.array(data)
                data1 = hf.get('trainlabel')
                trainlabel = numpy.array(data1)
                data2 = hf.get('testdata')
                testdata = numpy.array(data2)
                data3 = hf.get('testlabel')
                testlabel = numpy.array(data3)
                
                numpy.random.seed(100)
                numpy.random.shuffle(traindata)
                
                numpy.random.seed(100)
                numpy.random.shuffle(trainlabel)
                
                train_set = (traindata/numpy.max(traindata), trainlabel)
                test_set = (testdata/numpy.max(testdata), testlabel)
            
    else:
        try:
            with gzip.open(dataset, 'rb') as f:
                train_set, valid_set, test_set = pickle.load(f)
        except:    
            with gzip.open(dataset, 'rb') as f:
                train_set, test_set = pickle.load(f)
            f.close()
            print(len(train_set[0]), len(test_set[0]))
            traindata = train_set[0]
            trainlabel = train_set[1]   
            
            numpy.random.seed(100)
            numpy.random.shuffle(traindata)
            
            numpy.random.seed(100)
            numpy.random.shuffle(trainlabel)
            
            train_set = (traindata, trainlabel)      
            if (val):            
                n_valid = int(round(0.2 * len(traindata)))                
                               
                valid_data = traindata[len(traindata)-n_valid:len(traindata)]
                
                valid_label = trainlabel[len(trainlabel)-n_valid:len(trainlabel)]
                
                valid_set = (valid_data, valid_label)
                
                train_set = (traindata[0:len(traindata)-n_valid],trainlabel[0:len(trainlabel)-n_valid])
                print(len(train_set[0]), len(valid_set[0]),len(test_set[0]))                            

    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    
    if(trainShared):
        train_set_x, train_set_y = shared_dataset(train_set)
    else:
        train_set_x, train_set_y = train_set
        
        
    rval = [(train_set_x, train_set_y), (test_set_x, test_set_y)]
    
    if(val):
        valid_set_x, valid_set_y = shared_dataset(valid_set)
        rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
            
    return rval