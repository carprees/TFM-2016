# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 11:08:16 2016

@author: Carprees
"""
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal.pool import pool_2d as downsample
from theano.tensor.nnet import conv2d

import Utilidades.Util as ut
import Utilidades.RNA_util as rna

class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, isTrain, isBNTrain, image_shape, filter_shape, poolsize=None,
                 padding=(1,1), stride=(1,1), activation=rna.tanh(), batch_norm= False, p_drop = 0, gaussian_std=0.):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        
        if(poolsize==None):
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]))
            
        else:
            # each unit in the lower layer receives a gradient from:
            # "num output feature maps * filter height * filter width" /
            #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) //
                       numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        conv_out = conv2d(
            input=self.input,
            filters=self.W,
            filter_shape=filter_shape,
            input_shape=image_shape,
            border_mode=padding,
            subsample=stride
            )

        # downsample each feature map individually, using maxpooling
        if(poolsize==None):
            self.lin_output = conv_out
            
            self.im_dim_out = [(image_shape[2]-filter_shape[2]+1+padding[0]*2)/stride[0], (image_shape[3]-filter_shape[3]+1+padding[1]*2)/stride[1]]
            
        else:    
            pooled_out = downsample(
                input=conv_out,
                ds=poolsize,
                ignore_border=True,
                mode='max'
            )
            
            self.lin_output = pooled_out
           
            self.im_dim_out = [(image_shape[2]-filter_shape[2]+1+padding[0]*2)/poolsize[0], (image_shape[3]-filter_shape[3]+1+padding[1]*2)/poolsize[1]]
       
        self.n_im_out = filter_shape[0]
            
        self.n_pix_out = filter_shape[0] * self.im_dim_out[0] * self.im_dim_out[1]
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        
        if batch_norm:                                                 
            self.gamma = theano.shared(
                numpy.ones((filter_shape[0],), dtype=theano.config.floatX),
                borrow=True
            )     
            
            self.beta = theano.shared(
                numpy.zeros((filter_shape[0],), dtype=theano.config.floatX),
                borrow=True
            )      
            
            self.params = [self.W, self.gamma, self.beta]
            
            self.train_mean = theano.shared(
                numpy.zeros((filter_shape[0], self.im_dim_out[0],self.im_dim_out[1],), dtype=theano.config.floatX),
                borrow=True
            )  
            
            self.train_std = theano.shared(
                numpy.zeros((filter_shape[0], self.im_dim_out[0], self.im_dim_out[1],), dtype=theano.config.floatX),
                borrow=True
            ) 
            
            self.BN_params = [self.train_mean, self.train_std]            
            
            self.lin_output = rna.gaussian_perturb(self.lin_output, gaussian_std)
            
            self.mean = self.lin_output.mean((0,), keepdims=False)
            
            self.std = self.lin_output.std((0,), keepdims = False)            
                                      
            bn_output = T.nnet.bn.batch_normalization(self.lin_output, self.gamma.dimshuffle('x', 0, 'x', 'x'), 
                                                                self.beta.dimshuffle('x',0,'x', 'x'), 
                                                               mean = theano.ifelse.ifelse(isBNTrain, self.mean,  self.train_mean),
                    std = theano.ifelse.ifelse(isBNTrain, self.std, self.train_std), mode='high_mem')
            
#            bn_output = T.nnet.bn.batch_normalization(lin_output, self.gamma.dimshuffle('x',0, 'x', 'x'), 
#                                                                self.beta.dimshuffle('x',0, 'x', 'x'), 
#                                                               mean = lin_output.mean((0,), keepdims=True),
#                    std = lin_output.std((0,), keepdims = True) , mode='low_mem')

            self.output = rna.dropout_conv((
                bn_output if activation is None
                else (T.clip(bn_output,0,20) if activation is T.nnet.relu else activation(bn_output))
            ), isTrain, p_drop, filter_shape[0], im_shape=image_shape[0])

            
        else:
            self.lin_output = rna.gaussian_perturb(self.lin_output, gaussian_std) + self.b.dimshuffle('x', 0, 'x', 'x')
            self.output = rna.dropout_conv((
                self.lin_output if activation is None
                else activation(self.lin_output)
            ), isTrain, p_drop, filter_shape[0], im_shape=image_shape[0])
            # parameters of the model
            self.params = [self.W, self.b] 

        # keep track of model input
        self.input = input

class ResidualLayer(object):
    def __init__(self, rng, input, isTrain, isBNTrain, image_shape, filter_shape, padding=(1,1), stride=(1,1), pool = False, Residual = True,
                 activation=rna.tanh(), batch_norm = False, p_drop=0, gaussian_std=0.):
        """
        Allocate a ResidualLayer with shared variable internal parameters.
        """
        self.input = input
        
        self.ResidualLayer=[ConvLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=image_shape,
            filter_shape=filter_shape,
            padding=padding,
            stride=stride,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop,
            gaussian_std=gaussian_std
        )]
        
        if(Residual):
            self.ResidualLayer.append(ConvLayer(
                rng,
                input=self.ResidualLayer[0].output,
                isTrain=isTrain,
                isBNTrain=isBNTrain,
                image_shape=(image_shape[0], filter_shape[0], self.ResidualLayer[0].im_dim_out[0], self.ResidualLayer[0].im_dim_out[0]),
                filter_shape=(filter_shape[0], filter_shape[0], filter_shape[2], filter_shape[3]),
                padding=padding,
                activation=None,
                batch_norm=batch_norm,
                p_drop=p_drop,
                gaussian_std=gaussian_std
            ))
        else: 
            self.ResidualLayer.append(ConvLayer(
                rng,
                input=self.ResidualLayer[0].output,
                isTrain=isTrain,
                isBNTrain=isBNTrain,
                image_shape=(image_shape[0], filter_shape[0], self.ResidualLayer[0].im_dim_out[0], self.ResidualLayer[0].im_dim_out[0]),
                filter_shape=(filter_shape[0], filter_shape[0], filter_shape[2], filter_shape[3]),
                padding=padding,
                activation=activation,
                batch_norm=batch_norm,
                p_drop=p_drop,
                gaussian_std=gaussian_std
            ))
        
        self.params = ut.unlist([layer.params for layer in self.ResidualLayer])
        
        if(batch_norm):
            self.BN_params = ut.unlist([layer.BN_params for layer in self.ResidualLayer])
        
        if(Residual):
            output = activation(self.ResidualLayer[-1].output + self.input)
        else:
            output = self.ResidualLayer[-1].output
            
        if(pool):          
            self.output = downsample(input=output,ds=(2,2),ignore_border=True,mode='max')  
            
            self.n_pix_out = self.ResidualLayer[-1].n_pix_out/4
        
            self.im_dim_out = [self.ResidualLayer[-1].im_dim_out[0]/2, self.ResidualLayer[-1].im_dim_out[1]/2]
        else:
            self.output=output
            
            self.n_pix_out = self.ResidualLayer[-1].n_pix_out
        
            self.im_dim_out = [self.ResidualLayer[-1].im_dim_out[0], self.ResidualLayer[-1].im_dim_out[1]]
      
        
        
        self.n_im_out = self.ResidualLayer[-1].n_im_out

#class CNN(object):
#    """ Complete convolutional network """
#    
#    def __init__(self, rng, input, batch_size, im_dim=[], n_kerns=[20, 50], kerns_shape=[[5,5],[5,5]], poolsize=(2, 2)):
#        """
#        Allocate CNN with shared variable internal parameters built with different types of conv networks.
#
#        :type rng: numpy.random.RandomState
#        :param rng: a random number generator used to initialize weights
#
#        :type input: theano.tensor.dtensor4
#        :param input: symbolic image tensor, of shape image_shape
#
#        :type poolsize: tuple or list of length 2
#        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
#        """
#        assert len(n_kerns)==len(kerns_shape)
#        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
#        # to a 4D tensor, compatible with our LeNetConvPoolLayer
#        # (28, 28) is the size of MNIST images.
#        
#        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))
#        
#        # Construct the first convolutional pooling layer:
#        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
#        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
#        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
#        self.convLayers = [LeNetConvPoolLayer(
#            rng,
#            input=self.input,
#            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
#            filter_shape=(n_kerns[0], 1, kerns_shape[0][0], kerns_shape[0][1]),
#            poolsize = poolsize
#        )]
#
#        for i in range(1, len(n_kerns)):
#            self.convLayers.append(LeNetConvPoolLayer(
#                rng,
#                input=self.convLayers[i-1].output,
#                image_shape=(batch_size, n_kerns[i-1], self.convLayers[i-1].im_dim_out[0], self.convLayers[i-1].im_dim_out[1]),
#                filter_shape=(n_kerns[i], n_kerns[i-1], kerns_shape[i][0], kerns_shape[i][1]),
#                poolsize = poolsize
#            ))
#            
#        self.params = ut.unlist([convLayer.params for convLayer in self.convLayers])
#        
#        self.output = self.convLayers[-1].output
#        
#        self.n_out = self.convLayers[-1].n_im_out
 
class YichuanConv(object):
    """ Complete convolutional network """
    
    def __init__(self, rng, input, isTrain, isBNTrain, batch_size, im_dim=[], activation=rna.tanh(), 
                 batch_norm = False, p_drop=[0,0,0,0], gaussian_std=0.):
        """
        Allocate CNN with shared variable internal parameters built with different types of conv networks.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        # Reshape matrix of rasterized images of shape (batch_size, x * y)
        # to a 4D tensor.
        # (28, 28) is the size of MNIST images.
        # (48, 48) is the size of MNIST images.
        
        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))       
        
        # First convPool layer filters 64 3*3
        self.convLayers = [ConvLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
            filter_shape=(32, 1, 5, 5),
            poolsize = (2, 2), 
            padding=(2,2),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[0],
            gaussian_std=gaussian_std
        )]
        
        # Second convPool layer filters 128 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[0].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 32, self.convLayers[0].im_dim_out[0], self.convLayers[0].im_dim_out[1]),
            filter_shape=(32, 32, 4, 4),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))
        
       
        # Third conv + convPool layer filters 256 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[1].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 32, self.convLayers[1].im_dim_out[0], self.convLayers[1].im_dim_out[1]),
            filter_shape=(64, 32, 5, 5),
            poolsize = (2, 2),
            padding=(2,2),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=0.
        ))
        
        self.params = ut.unlist([layer.params for layer in self.convLayers])
        
        if(batch_norm):
            self.BN_params = ut.unlist([layer.BN_params for layer in self.convLayers])
        
        self.output = self.convLayers[-1].output
        
        self.n_im_out = self.convLayers[-1].n_im_out
        
        self.n_out = self.convLayers[-1].n_pix_out
    
class OxfordNet11LayerRed(object):
    """ Complete convolutional network """
    
    def __init__(self, rng, input, isTrain, isBNTrain, batch_size, im_dim=[], activation=rna.tanh(), 
                 batch_norm = False, p_drop=[0,0,0,0], gaussian_std=0.):
        """
        Allocate CNN with shared variable internal parameters built with different types of conv networks.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        # Reshape matrix of rasterized images of shape (batch_size, x * y)
        # to a 4D tensor.
        # (28, 28) is the size of MNIST images.
        # (48, 48) is the size of MNIST images.
        
        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))       
        
        # First convPool layer filters 64 3*3
        self.convLayers = [ConvLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
            filter_shape=(64, 1, 3, 3),
            poolsize = (2, 2), 
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[0],
            gaussian_std=gaussian_std
        )]
        
        # Second convPool layer filters 128 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[0].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[0].im_dim_out[0], self.convLayers[0].im_dim_out[1]),
            filter_shape=(128, 64, 3, 3),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))
        
       
        # Third conv + convPool layer filters 256 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[1].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 128, self.convLayers[1].im_dim_out[0], self.convLayers[1].im_dim_out[1]),
            filter_shape=(256, 128, 3, 3),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))
        
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[2].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 256, self.convLayers[2].im_dim_out[0], self.convLayers[2].im_dim_out[1]),
            filter_shape=(256, 256, 3, 3),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=gaussian_std
        ))
                       
            
        self.params = ut.unlist([layer.params for layer in self.convLayers])
        
        if(batch_norm):
            self.BN_params = ut.unlist([layer.BN_params for layer in self.convLayers])
        
        self.output = self.convLayers[-1].output
        
        self.n_im_out = self.convLayers[-1].n_im_out
        
        self.n_out = self.convLayers[-1].n_pix_out
        
class OxfordNet11Layer(object):
    """ Complete convolutional network """
    
    def __init__(self, rng, input, isTrain, isBNTrain, batch_size, im_dim=[], activation=rna.tanh(), 
                 batch_norm = False, p_drop=[0,0,0,0], gaussian_std=0.):
        """
        Allocate CNN with shared variable internal parameters built with different types of conv networks.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """
        # Reshape matrix of rasterized images of shape (batch_size, x * y)
        # to a 4D tensor.
        # (28, 28) is the size of MNIST images.
        # (48, 48) is the size of Emotions images.
        
        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))       
        
        # First convPool layer filters 64 3*3
        self.convLayers = [ConvLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
            filter_shape=(64, 1, 3, 3),
            poolsize = (2, 2), 
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[0],
            gaussian_std=gaussian_std
        )]
        
        # Second convPool layer filters 128 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[0].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[0].im_dim_out[0], self.convLayers[0].im_dim_out[1]),
            filter_shape=(128, 64, 3, 3),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))
        
       
        # Third conv + convPool layer filters 256 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[1].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 128, self.convLayers[1].im_dim_out[0], self.convLayers[1].im_dim_out[1]),
            filter_shape=(256, 128, 3, 3),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))
        
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[2].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 256, self.convLayers[2].im_dim_out[0], self.convLayers[2].im_dim_out[1]),
            filter_shape=(256, 256, 3, 3),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=0.
        ))
        
        # Fourth conv + convPool layer filters 512 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[3].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 256, self.convLayers[3].im_dim_out[0], self.convLayers[3].im_dim_out[1]),
            filter_shape=(512, 256, 3, 3),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[4],
            gaussian_std=gaussian_std
        ))
        
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[4].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 512, self.convLayers[4].im_dim_out[0], self.convLayers[4].im_dim_out[1]),
            filter_shape=(512, 512, 3, 3),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[5],
            gaussian_std=0.
        ))
            
        self.params = ut.unlist([layer.params for layer in self.convLayers])
        
        if(batch_norm):
            self.BN_params = ut.unlist([layer.BN_params for layer in self.convLayers])
        
        self.output = self.convLayers[-1].output
        
        self.n_im_out = self.convLayers[-1].n_im_out
        
        self.n_out = self.convLayers[-1].n_pix_out
             
class ResidualNet18Layer(object):
    """ Complete convolutional network """
    
    def __init__(self, rng, input, isTrain, isBNTrain, batch_size, im_dim=[], activation=rna.tanh(), 
                 batch_norm = False, p_drop=[0,0,0,0], gaussian_std=0.):
        """
        Allocate ResidualCNN with shared variable internal parameters built with different types of conv networks.

        """
        # Reshape matrix of rasterized images of shape (batch_size, x * y)
        # to a 4D tensor.
        # (28, 28) is the size of MNIST images.
        # (48, 48) is the size of Emotions images.
        
        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))       
        
        # First convPool layer filters 64 3*3
        self.convLayers = [ConvLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
            filter_shape=(64, 1, 5, 5),
            poolsize = (2, 2),
            padding=(2,2),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[0],
            gaussian_std=gaussian_std
        )]
        
        # 2 x Residual layer 64@3x3
        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[0].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[0].im_dim_out[0], self.convLayers[0].im_dim_out[1]),
            filter_shape=(64, 64, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))
       
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[1].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[1].im_dim_out[0], self.convLayers[1].im_dim_out[1]),
            filter_shape=(64, 64, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))
        
#        # 2 x Residual layer 128@3x3
        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[2].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[2].im_dim_out[0], self.convLayers[2].im_dim_out[1]),
            filter_shape=(128, 64, 3, 3),
            padding=(1,1),
            stride=(2,2),
            Residual = False,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))
       
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[3].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 128, self.convLayers[3].im_dim_out[0], self.convLayers[3].im_dim_out[1]),
            filter_shape=(128, 128, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))

        # 2 x Residual layer 256@3x3
        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[4].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 128, self.convLayers[4].im_dim_out[0], self.convLayers[4].im_dim_out[1]),
            filter_shape=(256, 128, 3, 3),
            padding=(1,1),
            stride=(2,2),
            Residual = False,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=gaussian_std
        ))
       
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[5].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 256, self.convLayers[5].im_dim_out[0], self.convLayers[5].im_dim_out[1]),
            filter_shape=(256, 256, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=gaussian_std
        ))
        

        self.params = ut.unlist([layer.params for layer in self.convLayers])
        
        if(batch_norm):
            self.BN_params = ut.unlist([layer.BN_params for layer in self.convLayers])
        
        self.output = self.convLayers[-1].output
        
        self.n_im_out = self.convLayers[-1].n_im_out
        
        self.n_out = self.convLayers[-1].n_pix_out
        
class ResidualNet18LayerRed(object):
    """ Complete convolutional network """
    
    def __init__(self, rng, input, isTrain, isBNTrain, batch_size, im_dim=[], activation=rna.tanh(), 
                 batch_norm = False, p_drop=[0,0,0,0], gaussian_std=0.):
        """
        Allocate ResidualCNN with shared variable internal parameters built with different types of conv networks.

        """
        # Reshape matrix of rasterized images of shape (batch_size, x * y)
        # to a 4D tensor.
        # (28, 28) is the size of MNIST images.
        # (48, 48) is the size of Emotions images.
        
        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))       
        
        # First convPool layer filters 32 3*3
        self.convLayers = [ConvLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
            filter_shape=(32, 1, 3, 3),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[0],
            gaussian_std=gaussian_std
        )]
        
        # 2 x Residual layer 32 @ 3 x 3
        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[0].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 32, self.convLayers[0].im_dim_out[0], self.convLayers[0].im_dim_out[1]),
            filter_shape=(32, 32, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))
       
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[1].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 32, self.convLayers[1].im_dim_out[0], self.convLayers[1].im_dim_out[1]),
            filter_shape=(32, 32, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=True,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))
        
#        # 2 x Residual layer 64@3x3
        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[2].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 32, self.convLayers[2].im_dim_out[0], self.convLayers[2].im_dim_out[1]),
            filter_shape=(64, 32, 3, 3),
            padding=(1,1),
            Residual = False,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))
       
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[3].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[3].im_dim_out[0], self.convLayers[3].im_dim_out[1]),
            filter_shape=(64, 64, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=True,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))

        # 2 x Residual layer 256@3x3
        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[4].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[4].im_dim_out[0], self.convLayers[4].im_dim_out[1]),
            filter_shape=(128, 64, 3, 3),
            padding=(1,1),
            Residual = False,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=gaussian_std
        ))
       
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[5].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 128, self.convLayers[5].im_dim_out[0], self.convLayers[5].im_dim_out[1]),
            filter_shape=(128, 128, 3, 3),
            padding=(1,1),
            Residual=True,
            pool=False,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=gaussian_std
        ))
        

        self.params = ut.unlist([layer.params for layer in self.convLayers])
        
#        if(batch_norm):
#            self.BN_params = ut.unlist([layer.BN_params for layer in self.convLayers])
        
        self.output = self.convLayers[-1].output
        
        self.n_im_out = self.convLayers[-1].n_im_out
        
        self.n_out = self.convLayers[-1].n_pix_out
        
class ResidualNetOxfordMod(object):
    """ Complete convolutional network """
    
    def __init__(self, rng, input, isTrain, isBNTrain, batch_size, im_dim=[], activation=rna.tanh(), 
                 batch_norm = False, p_drop=[0,0,0,0], gaussian_std=0.):
        """
        Allocate ResidualCNN with shared variable internal parameters built with different types of conv networks.

        """
        # Reshape matrix of rasterized images of shape (batch_size, x * y)
        # to a 4D tensor.
        # (28, 28) is the size of MNIST images.
        # (48, 48) is the size of Emotions images.
        
        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))  
        
        # First convPool layer filters 64 3*3
        self.convLayers = [ConvLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
            filter_shape=(64, 1, 3, 3),
            poolsize = (2, 2), 
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[0],
            gaussian_std=gaussian_std
        )]
        
        # Second convPool layer filters 128 3*3
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[0].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[0].im_dim_out[0], self.convLayers[0].im_dim_out[1]),
            filter_shape=(128, 64, 3, 3),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))

        # 2 x Residual layer 256@3x3        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[1].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 128, self.convLayers[1].im_dim_out[0], self.convLayers[1].im_dim_out[1]),
            filter_shape=(128, 128, 3, 3),
            padding=(1,1),
            pool = True,
            Residual = True,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))  

        self.params = ut.unlist([layer.params for layer in self.convLayers])
        
        if(batch_norm):
            self.BN_params = ut.unlist([layer.BN_params for layer in self.convLayers])
        
        self.output = self.convLayers[-1].output
        
        self.n_im_out = self.convLayers[-1].n_im_out
        
        self.n_out = self.convLayers[-1].n_pix_out

class MiniResidual(object):
    """ Complete convolutional network """
    
    def __init__(self, rng, input, isTrain, isBNTrain, batch_size, im_dim=[], activation=rna.tanh(), 
                 batch_norm = False, p_drop=[0,0,0,0], gaussian_std=0.):
        """
        Allocate ResidualCNN with shared variable internal parameters built with different types of conv networks.

        """
        # Reshape matrix of rasterized images of shape (batch_size, x * y)
        # to a 4D tensor.
        # (28, 28) is the size of MNIST images.
        # (48, 48) is the size of Emotions images.
        
        self.input = input.reshape((batch_size, 1, im_dim[0], im_dim[1]))  

        # 32 @ 5 x 5
        self.convLayers = [ResidualLayer(
            rng,
            input=self.input,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 1, im_dim[0], im_dim[1]),
            filter_shape=(32, 1, 3, 3),
            padding=(1,1),
            pool = True,
            Residual = True,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[0],
            gaussian_std=gaussian_std
        )] 
        
        # 32 @ 5 x 5
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[0].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 32, self.convLayers[0].im_dim_out[0], self.convLayers[0].im_dim_out[1]),
            filter_shape=(32, 32, 3, 3),
            padding=(1,1),
            pool = False,
            Residual = True,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[1],
            gaussian_std=gaussian_std
        ))  
               
        self.convLayers.append(ConvLayer(
            rng,
            input=self.convLayers[1].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 32, self.convLayers[1].im_dim_out[0], self.convLayers[1].im_dim_out[1]),
            filter_shape=(64, 32, 4, 4),
            poolsize = (2, 2),
            padding=(1,1),
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[2],
            gaussian_std=gaussian_std
        ))
        
        # Residual layer 64@5x5        
        self.convLayers.append(ResidualLayer(
            rng,
            input=self.convLayers[2].output,
            isTrain=isTrain,
            isBNTrain=isBNTrain,
            image_shape=(batch_size, 64, self.convLayers[2].im_dim_out[0], self.convLayers[2].im_dim_out[1]),
            filter_shape=(64, 64, 3, 3),
            padding=(1,1),
            pool = True,
            Residual = True,
            activation=activation,
            batch_norm=batch_norm,
            p_drop=p_drop[3],
            gaussian_std=gaussian_std
        ))  

        self.params = ut.unlist([layer.params for layer in self.convLayers])
        
        if(batch_norm):
            self.BN_params = ut.unlist([layer.BN_params for layer in self.convLayers])
        
        self.output = self.convLayers[-1].output
        
        self.n_im_out = self.convLayers[-1].n_im_out
        
        self.n_out = self.convLayers[-1].n_pix_out