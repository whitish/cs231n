from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = weight_scale * np.random.randn(input_dim, hidden_dim)
        b1 = np.zeros(hidden_dim)
        W2 = weight_scale * np.random.randn(hidden_dim, num_classes)
        b2 = np.zeros(num_classes)

        self.params.update(dict({'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # Forward into first layer
        hidden_layer, cache_hidden_layer = affine_relu_forward(X, self.params.get('W1'), self.params.get('b1'))
        # Forward into second layer
        scores, cache_scores = affine_forward(hidden_layer, self.params.get('W2'), self.params.get('b2'))

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        N = X.shape[0]  # X.shape[0] - number of observations in X
        scores -= np.max(scores, axis=1, keepdims=True)  # numeric stability
        probs = np.exp(scores)  # e^XW
        probs /= np.sum(probs, axis=1, keepdims=True)
        correct_probs = probs[np.arange(N), y]
        data_loss = -np.sum(np.log(correct_probs)) / N
        reg_loss = 0.5 * self.reg * np.sum(self.params.get('W1') ** 2)
        reg_loss += 0.5 * self.reg * np.sum(self.params.get('W2') ** 2)
        loss = data_loss + reg_loss

        # dx = DL_dscores - see layer.py (softmax_loss) with the same calculations
        DL_dscores = probs.copy()
        DL_dscores[np.arange(N), y] -= 1
        DL_dscores /= N

        # Backprop into second layer
        dx1, dW2, db2 = affine_backward(DL_dscores, cache_scores)
        dW2 += self.reg * self.params.get('W2')

        # Backprop into first layer
        dx, dW1, db1 = affine_relu_backward(dx1, cache_hidden_layer)
        dW1 += self.reg * self.params.get('W1')

        grads.update(dict({'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_batchnorm = normalization == 'batchnorm'
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        L = len(hidden_dims)
        if L < 2:
            ValueError("L must be >=2. If L = 1 - use class TwoLayerNetwork()")

        # for L >= 2
        # add params W, b for the 1st layer
        W1 = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        b1 = np.zeros(hidden_dims[0])
        self.params.update(dict({'W1': W1}))  # insert W1
        self.params.update(dict({'b1': b1}))  # insert b1


        if self.use_batchnorm:
            # add params for the 1st layer
            gamma1 = np.ones(hidden_dims[0])  # scale parameter for batch normalization
            beta1 = np.zeros(hidden_dims[0])  # shift parameter for batch normalization
            self.params.update(dict({'gamma1': gamma1}))  # insert gamma1
            self.params.update(dict({'beta1': beta1}))  # insert beta1

        gamma, beta, gamma_names, beta_names = [], [], [], []

        W, b, W_names, b_names = [], [], [], []

        for i in np.arange(2, L+1):  # for L >= 2
            W = weight_scale * np.random.randn(hidden_dims[i-2], hidden_dims[i-1])
            b = np.zeros(hidden_dims[i-1])
            if self.use_batchnorm:
                gamma = np.ones(hidden_dims[i-1])
                beta = np.zeros(hidden_dims[-1])
                gamma_names = 'gamma' + str(i)
                beta_names = 'beta' + str(i)
                self.params.update(dict({gamma_names: gamma}))
                self.params.update(dict({beta_names: beta}))


            W_names = 'W' + str(i)  # W2, W3, ...
            b_names = 'b' + str(i)  # b2, b3, ...

            self.params.update(dict({W_names : W}))
            self.params.update(dict({b_names : b}))

        WL = weight_scale * np.random.randn(hidden_dims[L-1], num_classes)
        bL = np.zeros(num_classes)

        self.params.update(dict({'W'+str(L+1): WL}))  # insert last W - WL
        self.params.update(dict({'b'+str(L+1): bL}))  # insert last b - bL

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = []  # list to keep cache of every layer
        dropout_cache = []

        hidden_layer = X

        for i in range(1, self.num_layers):  # forward process for layer 2, 3, ... N-1
            W = self.params.get('W'+str(i))
            b = self.params.get('b'+str(i))
            if self.use_batchnorm:
                gamma = self.params['gamma'+str(i)]
                beta = self.params['beta'+str(i)]
                # Forward pass into ReLU layers + batch_norm + apply dropout
                hidden_layer, cache_tmp = affine_bn_relu_forward(hidden_layer, W, b, gamma, beta, self.bn_params[i-1])
            else:
                # Forward pass into ReLU layers + apply dropout after each ReLU non linearity
                hidden_layer, cache_tmp = affine_relu_forward(hidden_layer, W, b)

            cache.append(cache_tmp)  # append cache of i-th layer

            # dropout after every ReLU layer
            if self.use_dropout:
                hidden_layer, drop_cache = dropout_forward(hidden_layer, self.dropout_param)  # dropout (if self.dropout_param['p'] > 0)!
                dropout_cache.append(drop_cache)

        # forward pass for the last (affine) layer
        W, b = self.params.get('W'+str(self.num_layers)), self.params.get('b'+str(self.num_layers))
        scores, cache_scores = affine_forward(hidden_layer, W, b)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        N = X.shape[0]  # X.shape[0] - number of observations in X
        scores -= np.max(scores, axis=1, keepdims=True)  # numeric stability
        probs = np.exp(scores)  # e^XW
        probs /= np.sum(probs, axis=1, keepdims=True)
        correct_probs = probs[np.arange(N), y]
        data_loss = -np.sum(np.log(correct_probs)) / N

        # cumulatively add reg loss
        reg_loss = 0
        for i in range(1, self.num_layers+1):
            reg_loss += 0.5 * self.reg * np.sum(self.params.get('W'+str(i)) ** 2)

        loss = data_loss + reg_loss

        # dx = DL_dscores - see layer.py (softmax_loss) with the same calculations
        DL_dscores = probs.copy()
        DL_dscores[np.arange(N), y] -= 1
        DL_dscores /= N

        # Backprop into the last (affine) layer
        # {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
        dxN, dWL, dbL = affine_backward(DL_dscores, cache_scores)
        dWL += self.reg * self.params.get('W'+str(self.num_layers))
        grads.update(dict({'W'+str(self.num_layers): dWL, 'b'+str(self.num_layers): dbL}))

        # Backprop into the all remaining (self.num_layers - 1) ReLU hidden layers iteratively
        for i in range(self.num_layers-1, 0, -1):

            # Dropout backprop before each ReLU non linearity
            if self.use_dropout:
                dxN = dropout_backward(dxN, dropout_cache[i-1])

            if self.use_batchnorm:
                dxN, dWN, dbN, dgammaN, dbetaN = affine_bn_relu_backward(dxN, cache[i-1])
            else:
                # Backprop into ReLU layer (after dropout backprop)
                dxN, dWN, dbN = affine_relu_backward(dxN, cache[i-1])

            dWN += self.reg * self.params.get('W'+str(i))

            grads.update(dict({'W'+str(i): dWN, 'b'+str(i): dbN}))

            if self.use_batchnorm:
                grads.update(dict({'gamma'+str(i): dgammaN, 'beta'+str(i): dbetaN}))
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
