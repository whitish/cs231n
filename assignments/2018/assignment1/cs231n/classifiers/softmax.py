import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]  # number of observations in X
  num_classes = W.shape[1]
  s = X.dot(W)  # scores
  s -= np.max(s)  # to make numbers more stable
  for obs in range(num_train):
    probs = np.exp(s[obs,:])  # e^XW
    probs_sum = np.sum(probs)
    probs_normalized = probs / probs_sum  # sum of all probs = 1
    loss += -1*np.log(probs_normalized[y[obs]])
    for k in range(num_classes):
        dW[:,k] += X[obs]*(probs_normalized[k] - (y[obs] == k))

  dW = dW/num_train + reg*W  # add regularization

  loss /= num_train
  loss += reg * np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]  # number of observations in X
  scores = X.dot(W)
  scores -= np.max(scores, axis=1, keepdims=True)  # numeric stability
  probs = np.exp(scores)  # e^XW
  probs = np.apply_along_axis(lambda t: t / np.sum(t), 1, probs)

  correct_probs = probs[np.arange(num_train), y]
  loss = np.sum(-1*np.log(correct_probs))

  loss /= num_train
  loss += reg * np.sum(W*W)

  ind = np.zeros_like(probs)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(probs - ind)

  dW /= num_train
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


class SoftmaxClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X_train, y_train, lr=1e-3, reg=1e-5, num_iters=100, batch_size=1000, verbose=False):
        # cross-validate over X_train: loss, dW
        dim = X_train.shape[1]
        num_classes = np.max(y_train) + 1
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run SGD to optimize W
        train_idx = np.arange(X_train.shape[0])
        loss_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None
            sample_idx = np.random.choice(train_idx, size = batch_size, replace = True)  # generate sample indexes for i-th iteration
            X_batch = X_train[sample_idx]
            y_batch = y_train[sample_idx]

            loss, grad = softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
            loss_history.append(loss)
            self.W = self.W - lr*grad

            if verbose and it % 10 == 0:
                print("iteration: %d/%d, loss: %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X_pred):
        XW = X_pred.dot(self.W)
        y_pred = np.argmax(XW, axis=1)  # return indices of maximum values along the axis

        return y_pred
