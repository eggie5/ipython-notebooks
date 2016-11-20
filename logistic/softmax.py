import numpy as np
  
class Softmax(object):
  """docstring for Softmax"""
  def __init__(self, arg):
    super(Softmax, self).__init__()
    self.arg = arg

  def softmax(self, x):
      """Compute softmax values for each sets of scores in x."""
     #should return a vector of probs
     # need the quotent to be a vector
      return (np.exp(x) / np.sum(np.exp(x), axis=1))

  def softmax2(self, x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out
    
  def softmax_T(self, x):
    return np.array(map(lambda vec: self.softmax2(vec), x))
    
  def softmax3(self, w):
    w = np.array(w)

    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1)

    return dist
    
  def softmax4(self, x):
      """Compute softmax values for each sets of scores in x."""
      e_x = np.exp(x - np.max(x))
      return e_x / e_x.sum(axis=1) # only difference

  def softmax5(self, w):
    w = np.array(w)

    maxes = np.amax(w, axis=1)
    maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e, axis=1)

    return dist

# [[ 1437.7  1123.6   481.9]
#  [ 1000.1   841.8   407.3]
#  [ 1541.6  1202.9   507.6]]

# x = np.array([[1, 2, 3],[4, 5, 9]])
# # x=np.random.randint(5, size=(3,3))
# print x
# probs = Softmax("").softmax_T(x)
# print probs
# print probs.sum(axis=1)

