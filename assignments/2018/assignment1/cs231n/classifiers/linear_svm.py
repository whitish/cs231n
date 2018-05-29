import numpy as np
from random import shuffle

# my implementation of gradient calculation
def my_svm_loss(W, X, y, reg, delta = 1):
    dL = np.zeros(W.shape)
    loss = 0.0

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):  # loop through the observations
        scores = X[i].dot(W)  # scores for i-th observation
        score_correct = scores[y[i]]  # score for correct class in i-th observation
        margin_positive_times = 0  # number of times when margin is positive
        for j in range(num_classes):  # calculate margin for every j-th class
            if j == y[i]:
                continue
            margin = scores[j] - score_correct + delta
            if margin > 0:
                loss += margin
                margin_positive_times += 1
                dL[:,j] += X[i]  # gradient dL/dW where j != y[i]
        dL[:,y[i]] += -1*X[i]*margin_positive_times

    dL /= X.shape[0]  # Average by number of observations
    dL += 2*reg*W  # add regularization

    loss /= X.shape[0]
    loss += reg * np.sum(W * W)  # add regularization

    return loss, dL



def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i].T  # partial derivative of loss by Wij
        dW[:,y[i]] -= X[i].T


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg, delta=1):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  scores = X.dot(W)
  correct_scores = np.array([scores[i,j] for (i,j) in zip(range(scores.shape[0]), y)])
  margin0 = (scores.T - correct_scores).T + delta  # for every observation deduct correct score
  margin = np.where(margin0 > 0, margin0, 0)  # get max(0, margin)
  margin = np.sum(margin, axis=1) - delta  # calculate loss for every observation as sum of margins.
  loss = sum(margin)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  num_train = X.shape[0]
  for i in range(num_train):  # loop through the observations
      #t = np.where(margin0[i,:] <= 0, 0, np.where(margin0[i,:] == 1, -1*X[i,:]*(sum(margin0[i,:]>0)-1)), X[i,:]))

      t = np.tile(X[i,:],(W.shape[1],1))
      t[y[i]] = -1*X[i,:]*(sum(margin0[i,:]>0)-1)
      t[np.where(margin0[i,:] <= 0)] = 0

      dW += t.T  # gradient dL/dW where j != y[i]


  dW /= X.shape[0]  # Average by number of observations
  dW += 2*reg*W  # add regularization
  
  loss /= X.shape[0]

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
