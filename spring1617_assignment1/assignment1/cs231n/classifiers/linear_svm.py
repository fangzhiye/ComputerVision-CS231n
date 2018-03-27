import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).#说明函数的功能

  Inputs have dimension D, there are C classes, and we operate on minibatches #说明输入
  of N examples.

  Inputs:#说明输入的数据类型
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W #返回W的梯度
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero #初始梯度矩阵为dW

  # compute the loss and the gradient
  # dw = (f(x+h)-f(x)) / h 
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
        dW[:,j] += X[i,:].T#每一样本进行计算的过程中，W列都需要加X[i,:]
        dW[:,y[i]] += -X[i,:].T
        loss += margin
  # margin = max{0,Sj - Syi + 1}
  # loss函数其实是X[x0 x1 x2 x3] W[w0 w1 w2 w3]  score = w0x0+w1x1+w2x2+w3x3
  #                               ...
  # 事实上如果展开margin的话 可以发现w0*x0即w0只和x0相关，那么w0的梯度其实就是x0
  # 事实W第一行的梯度（除了yi列外都是x0)
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg*W #梯度也要正则化吗？
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  # 计算损失函数的梯度并存在dW,在计算梯度的时                                    #                                     #                                                                           #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
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
  #X[N*D]*W[D*C] = N*C可以算出一样本每 
  scores = X.dot(W)
  num_classes = W.shape[1]
  num_train = X.shape[0]
  
  #np.array([:m],[:n])都是开区间，即得到array[0,m)个元素，再得到每个元素[0,n)个元素
  scores_correct = scores[np.arange(num_train),y] #得到第i个样本，y列的分数
  scores_correct = np.reshape(scores_correct,(num_train,-1))#将num_train个元素按列排
  margins = scores - scores_correct + 1 #一矩阵减一列，其实矩阵每列的元素会减去该列的对应元素
  margins = np.maximum(0,margins)#将小于的0的元素设置为0，第一个参数是什么就会设置为什么
  margins[np.arange(num_train),y] = 0#?这设置为0，真正类别的分数设为0，要不现在为1,其实真正类别是没有loss的
  loss += np.sum(margins) / num_train
  loss += 0.5 * reg * np.sum(W*W)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #如何来计算梯度呢？事实上
  #对天dW的每个元素，其实如N*D D*C D的元素如D11 dW11就是累加所有非1类的x0-所有1类的x0
  #其实对于X中只计算分数>正确类别的分数
  margins[margins > 0] = 1
  row_sum = np.sum(margins, axis = 1) #矩阵向量过程中0，1矩阵是一很好的工具
  margins[np.arange(num_train), y] = -row_sum #有几个是该类就要减去其x0
  dW += np.dot(X.T, margins)/num_train + reg * W
  #X.T转置之后第一行都是x0，margins（第一列中有0 1 -m)m就是类别是c1的个数
  #以dW11为例，累加所有分类大于0的x0-所有是c1的x0
  #如果某一样本，在某一类别中的得分损失函数小于0之后不考虑设为0即可
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
