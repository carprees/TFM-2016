# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 10:28:47 2016

@author: Carprees
"""
from __future__ import print_function

"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""

__docformat__ = 'restructedtext en'

import numpy

import theano
import theano.tensor as T

import Utilidades.Util as ut
import Utilidades.RNA_util as rna

class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, batch_norm = False):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.linear = T.dot(input, self.W) + self.b  
        
        if batch_norm:    
            self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma')
            self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta')
            
            self.params = [self.W, self.b, self.gamma, self.beta]
            
            self.bn_output = rna.logReg_batch_norm(self.linear, self.gamma, self.beta)
            
#            self.bn_output = T.sum(rna.batch_norm(self.linear, self.gamma, self.beta), axis = 0)
              
            self.p_y_given_x = T.nnet.softmax(self.bn_output)
        
        else:   

            self.p_y_given_x = T.nnet.softmax(self.linear) 
            
            # parameters of the model
            self.params = [self.W, self.b]
            
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
#        self.y_pred = T.argmax(self.linear, axis=1)
        # end-snippet-1


        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        # end-snippet-2

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
            
    def entropy(self, y):
        return T.mean(T.nnet.categorical_crossentropy(self.p_y_given_x, y))
        
    def hinge(self, u):
       return T.maximum(0, 1 - u)

    def ova_svm_cost(self, y1):
        """ return the one-vs-all svm cost
        given ground-truth y in one-hot {-1, 1} form """
        y1_printed = theano.printing.Print('this is important')(T.max(y1))
        margin = y1 * self.linear
        cost = self.hinge(margin).mean(axis=0).sum()
        return cost

# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, isTrain, isBNTrain, n_in, n_out, W=None, b=None,
                 activation = rna.tanh(), batch_norm = False, p_drop = 0.0, gaussian_std = 0.):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        
        if(batch_norm):    
            self.gamma = theano.shared(value = numpy.ones((n_out,), dtype=theano.config.floatX), name='gamma')
            self.beta = theano.shared(value = numpy.zeros((n_out,), dtype=theano.config.floatX), name='beta')
            
            self.params = [self.W, self.b, self.gamma, self.beta]
            
            self.train_mean = theano.shared(
                numpy.zeros((n_out,), dtype=theano.config.floatX),
                borrow=True
            )  
            
            self.train_std = theano.shared(
                numpy.zeros((n_out,), dtype=theano.config.floatX),
                borrow=True
            ) 
            
            self.BN_params = [self.train_mean, self.train_std]
            
            lin_output = T.dot(rna.gaussian_perturb(input, gaussian_std), self.W) + self.b
            
            self.mean = lin_output.mean((0,), keepdims=False)
            
            self.std = lin_output.std((0,), keepdims = False) 
            
            bn_output = T.nnet.bn.batch_normalization(lin_output, self.gamma, 
                                                    self.beta, 
                                                   mean = theano.ifelse.ifelse(isBNTrain, self.mean,  self.train_mean),
                                                std = theano.ifelse.ifelse(isBNTrain, self.std, self.train_std), mode='high_mem')

#            bn_output = T.nnet.bn.batch_normalization(lin_output, self.gamma, 
#                                                    self.beta, 
#                                                   mean = lin_output.mean((0,), keepdims=True),
#                                                std = lin_output.std((0,), keepdims = True) , mode='high_mem')
            
            self.output = rna.dropout((
                bn_output if activation is None
                else (T.clip(bn_output,0,20) if activation is T.nnet.relu else activation(bn_output))
            ), isTrain, p_drop)
            
        else:
            lin_output = T.dot(rna.gaussian_perturb(input, gaussian_std), self.W) + self.b
            self.output = rna.dropout((
                lin_output if activation is None
                else activation(lin_output)
            ), isTrain, p_drop)
            # parameters of the model
            self.params = [self.W, self.b]

        
# start-snippet-2
class deepMLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, isTrain, isBNTrain, n_in, n_hidden, n_out, activation=rna.tanh(), 
                 batch_norm = False,p_drop=0., gaussian_std=0.):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int or list deppending on number of hidden layers
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """
        if (isinstance(n_hidden,int)):
            # Since we are dealing with a one hidden layer MLP, this will translate
            # into a HiddenLayer with a tanh activation function connected to the
            # LogisticRegression layer; the activation function can be replaced by
            # sigmoid or any other nonlinear function
            self.hiddenLayer = HiddenLayer(
                rng=rng,
                input=input,
                isTrain=isTrain,
                isBNTrain=isBNTrain,
                n_in=n_in,
                n_out=n_hidden,
                activation = activation,
                batch_norm = batch_norm,
                p_drop = p_drop,
                gaussian_std=gaussian_std
            )
            
            # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayer.output,
                n_in=n_hidden,
                n_out=n_out,
                batch_norm=False
            )
            
            # end-snippet-2 start-snippet-3
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 = (
                abs(self.hiddenLayer.W).sum()
                + abs(self.logRegressionLayer.W).sum()
            )
    
            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2_sqr = (
                (self.hiddenLayer.W ** 2).sum()
                + (self.logRegressionLayer.W ** 2).sum()
            )
            
            # the parameters of the model are the parameters of the two layer it is
            # made out of
            self.params = self.hiddenLayer.params + self.logRegressionLayer.params
            # end-snippet-3
            if(batch_norm):
                self.BN_params = self.hiddenLayer.BN_params
            
        else:     
            self.hiddenLayers = [HiddenLayer(
                rng=rng,
                input=input,
                isTrain=isTrain,
                isBNTrain=isBNTrain,
                n_in=n_in,
                n_out=n_hidden[0],
                activation = activation,
                batch_norm = batch_norm,
                p_drop = p_drop,
                gaussian_std=gaussian_std
            )]
            
            for i in range(1, len(n_hidden)):
                self.hiddenLayers.append(HiddenLayer(
                    rng=rng,
                    input=self.hiddenLayers[i-1].output,
                    isTrain=isTrain,
                    isBNTrain=isBNTrain,
                    n_in=n_hidden[i-1],
                    n_out=n_hidden[i],
                    activation = activation,
                    batch_norm = batch_norm,
                    p_drop = p_drop,
                    gaussian_std=gaussian_std
                ))
            # The logistic regression layer gets as input the hidden units
            # of the hidden layer
            self.logRegressionLayer = LogisticRegression(
                input=self.hiddenLayers[-1].output,
                n_in=n_hidden[-1],
                n_out=n_out,
                batch_norm = False
            )
            
            # end-snippet-2 start-snippet-3
            # L1 norm ; one regularization option is to enforce L1 norm to
            # be small
            self.L1 = (
                sum([abs(hl.W).sum() for hl in self.hiddenLayers])
                + abs(self.logRegressionLayer.W).sum()
            )
    
            # square of L2 norm ; one regularization option is to enforce
            # square of L2 norm to be small
            self.L2_sqr = (
                sum([(hl.W ** 2).sum() for hl in self.hiddenLayers])
                + (self.logRegressionLayer.W ** 2).sum()
            )
            
            # the parameters of the model are the parameters of the two layer it is
            # made out of
            self.params = ut.unlist([hl.params for hl in self.hiddenLayers]) + self.logRegressionLayer.params
            
            if(batch_norm):
                self.BN_params = ut.unlist([layer.BN_params for layer in self.hiddenLayers])



        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        
        self.ova_svm_cost = self.logRegressionLayer.ova_svm_cost
        
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors
        
        self.entropy = self.logRegressionLayer.entropy

        # keep track of model input
        self.input = input
