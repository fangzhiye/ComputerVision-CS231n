import numpy as np
from past.builtins import xrange


class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=0):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test): #测试样本为i
      for j in xrange(num_train):#训练样本为j
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #dists[i,j]表示样本i到样本j的距离
        dists[i,j] = np.sqrt(np.sum(np.square(self.X_train[j,:] - X[i,:])))
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.
    使用单层循环来提升效率
    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train))
    for i in xrange(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      #应该是两个向量相乘
      #python中一个矩阵减一个向量，矩阵的每一行都会减去这个向量
      #np.square(矩阵）矩阵中的每个元素都平方
      #np.sum会按列或行将矩阵元素求和
        dists[i,:] = np.sqrt(np.sum(np.square(self.X_train - X[i,:]), axis = 1))
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]
    dists = np.zeros((num_test, num_train)) 
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    #主要是利用了公式（a-b)^2 = a^2-2ab+b^2
    #sqrt[(a-b)^2 + [c-d]^2 + [e-f]^2] = sqrt(a^2 + c^c + e^2 +b^2 + d^2 + f^2 -2ab -2cd -2ef)
    dists = np.multiply(np.dot(X, self.X_train.T),-2)
    #[5*100]*[100*10] = [5*10]每个元素都是-2ab，ab是100个元素点dot的和
    sq1 = np.sum(np.square(X),axis = 1, keepdims = True)
    #得到是[5*1]的列向量，都是每个像素平方和,keepdims是保证二维特性
    sq2 = np.sum(np.square(self.X_train), axis = 1)#得到的是[10*1]的列向量，都是每个像素平方各
    dists = np.add(dists,sq1)
    dists = np.add(dists,sq2)
    dists = np.sqrt(dists)
    #np.add如果shape不一样会转化为一样
    #sq1之所以保留其二维特性就是会按列复制向量构成矩阵，如果只是一向量，就按行复制得到矩阵
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.
    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in xrange(num_test):
      # A list of length k storing the labels of the k nearest neighbors to
      # the ith test point.
        closest_y = []
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################
        
        closest_index = np.argsort(dists[i])
        for j in range(k):
            closest_y.append(self.y_train[closest_index[j]])
        
        #closest_y = self.y_train[np.argsort(dists[i])[:k]]#取前k个
        #先利用argsort进行升序排序argsort(array)[:k]后取前个，返回的是索引
      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################

        #训练集里的元素并不是str的label而是数字
        #np.bincount会计算从0至数组中最大元素每个值出现的次数，这样np.bincount(数组)里的数组是不能有负数的
        #y_pred[i] = np.argmax(np.bincount(closest_y)) 
        #返回最大值的索引，axis=0是列，而axis = 1是行
        
        predDict = {}
        preDict = {label : closest_y.count(label) for label in set(closest_y)}
        result = sorted(preDict.items(), key=lambda item:item[1], reverse= True)
        #排序后返回一个数组[('key':value),('key':value),...]
        #最后要从返回值中获得对于turple要获得key 要result[0][0]这样
        y_pred[i] = result[0][0]
       
      #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

