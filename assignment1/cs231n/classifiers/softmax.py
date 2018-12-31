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
  
  num_train=X.shape[0]
  num_classes=W.shape[1]

  for i in range(num_train):
    
    scores=X[i].dot(W)  #scores is a 1 by C matrix    
    newscores=scores-np.max(scores)
    
    loss_i=np.log(sum(np.exp(newscores)))-newscores[y[i]]
    loss+=loss_i
    for j in range(num_classes):
      softmax_out=np.exp(newscores[j])/np.sum(np.exp(newscores))
      dW[:,j]+=(softmax_out -(j==y[i]))*X[i]

  loss=loss/num_train + 0.5*reg*np.sum(W*W)
  dW=dW/num_train +reg*W


  ''' 
  (N, D) = X.shape
  C = W.shape[1]
  #遍历每个样本
  for i in range(N):
    f_i = X[i].dot(W)
    #进行公式的指数修正
    f_i -= np.max(f_i)
    sum_j = np.sum(np.exp(f_i))
    #得到样本中每个类别的概率
    p = lambda k : np.exp(f_i[k]) / sum_j
    loss += - np.log(p(y[i]))
    #根据softmax求导公式
    for k in range(C):
      p_k = p(k)
      dW[:, k] += (p_k - (k == y[i])) * X[i]

  loss=loss/N + 0.5*reg*np.sum(W*W)
  dW=dW/N +reg*W
  '''
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train=X.shape[0]
  num_classes=W.shape[1]

  scores=X.dot(W) #N*C
  newscores=scores- np.max(scores,axis=1).reshape(-1,1)  
  softmax_out=np.exp(newscores)/ np.sum(np.exp(newscores),axis=1).reshape(-1,1)  #sotmax_out=N*C - N,=N*C
  loss=-np.sum(np.log(softmax_out[range(num_train),y]))
  loss=loss/num_train +0.5*reg*np.sum(W*W)

  ds=softmax_out.copy()
  ds[range(num_train),y]+=-1

  dW=(X.T).dot(ds)
  dW=dW/num_train +reg*W
  return loss, dW

