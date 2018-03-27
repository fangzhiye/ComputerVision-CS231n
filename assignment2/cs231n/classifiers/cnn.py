from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        C,H,W = input_dim
        self.params['W1'] = np.random.randn(num_filters,C,filter_size,filter_size)*weight_scale
        self.params['b1'] = np.zeros(num_filters)
        #经过卷积 池化之后的维度是是之前的1/4因为我是用最大池化1/2* 1/2 = 1/4
        #每一个卷积其实可以看成是一个神经元
        #N * num_filters * H * W /4
        self.params['W2'] = np.random.randn(int(num_filters * H * W /4), hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.randn(hidden_dim, num_classes)*weight_scale
        self.params['b3'] = np.zeros(num_classes)
        self.params['gamma'] = np.ones(num_filters)
        self.params['beta'] = np.zeros(num_filters)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        gamma,beta=self.params['gamma'],self.params['beta']  
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        bn_param={'mode':'test' if  y is None else 'train'} 
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        self.cache = {}
        y1, self.cache['cache1'] = conv_bn_relu_pool_forward(X,W1,b1,gamma,beta,conv_param,bn_param,pool_param)
        y2, self.cache['cache2'] = affine_relu_forward(y1,W2,b2)
        scores,self.cache['cache3'] = affine_forward(y2,W3,b3)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss,dy=softmax_loss(scores,y)  
        loss+=0.5*self.reg*(np.sum(np.square(self.params['W1']))+np.sum(np.square(self.params['W2']))+np.sum(np.square(self.params['W3'])))  
        grad_term,grads['W3'],grads['b3']=affine_backward(dy,self.cache['cache3'])  
        grads['W3']+=self.reg*self.params['W3']  
        grad_term,grads['W2'],grads['b2']=affine_relu_backward(grad_term,self.cache['cache2'])  
        grads['W2']+=self.reg*self.params['W2']  
        dx,grads['W1'],grads['b1'],grads['gamma'],grads['beta']=conv_bn_relu_pool_backward(grad_term,self.cache['cache1'])  
        grads['W1']+=self.reg*self.params['W1']  
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads