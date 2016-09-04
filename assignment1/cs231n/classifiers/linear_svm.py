import numpy as np
from random import shuffle

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
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:, j] += X[i]
        dW[:, y[i]] -= X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  scores_vec = np.dot(X, W)
  num_train = scores_vec.shape[0]
  num_classes = scores_vec.shape[1]
  correct_vec = scores_vec[range(0, num_train), y]#Get correct class score for every training example
  res_vec_ori = scores_vec - correct_vec.reshape(correct_vec.shape[0], 1) + 1
  res_vec_ori[range(0, num_train), y] = 0
  res_vec = res_vec_ori.copy()
  res_vec[res_vec<0] = 0#max(0, x)
  loss_vec = np.sum(res_vec, axis=1)
  loss = np.sum(loss_vec)
  loss /= num_train
    
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  pass
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
  d_res_vec = res_vec_ori.copy()
  d_res_vec[d_res_vec>0] = 1
  d_res_vec[d_res_vec<0] = 0
  sum_vec = np.sum(d_res_vec, axis=1)
  d_res_vec[range(0, num_train), y] = -sum_vec
  dW = np.dot(X.T, d_res_vec)
  dW /= num_train

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
