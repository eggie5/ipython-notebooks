from sklearn.linear_model import SGDClassifier
import softmax as sm
from sklearn import preprocessing

class GradientDescent(object):
  """GradientDescent"""
  def __init__(self, arg):
    super(GradientDescent, self).__init__()
    self.arg = arg
    self.W = np.array([]) #matrix of weights
    self.b = 0 #bias
    self.classes = []
    
  def fit(self, X, Y):
    #TODO Implement custom GD routine
    #hacking sickit to fill in GD routine until I write my own
    clf = SGDClassifier(loss="log", penalty="l2")
    clf.fit(X, Y)
    self.W = clf.coef_ 
    return self
  
  def logit(self, X):
    scores = X.dot(self.W.T) + self.b
    return scores#.ravel() if scores.shape[1] == 1 else scores
    
  def softmax(self, scores):
    return sm.Softmax("").softmax_T(scores)
    
  def predict_proba(self, X):
    #get logit scores
    scores = self.logit(X)
    #then feed to softmax to get probabilties
    probs = self.softmax(scores)
    
    return probs
    
  def predict_log_proba(self, X):
    off = 1e-6
    return -np.log(self.predict_proba(X)+off)
    
  def loss_function(self, x, y_true):
    #log-loss/cross-entropy
    y_log_prob = self.predict_log_proba(x)
    lb = preprocessing.LabelBinarizer().fit(self.classes)
    transformed_labels = lb.transform(y_true)
    loss = (transformed_labels * y_log_prob)#.sum(axis=1)

    return loss#np.average(loss)
    
  def fit2(self, X, Y):
    nb_epochs = 200
    params = []
    learning_rate = .1
    
    self.X = X
    self.Y = Y
    
    #get num of classes
    self.classes = np.unique(Y)
    
    #random init weights
    self.W = 0.01 * np.random.randn(len(self.classes),X.shape[1]) #4x3
    self.B = np.zeros((1,len(self.classes)))
    
    for i in range(nb_epochs):
      loss = self.loss_function(X, Y) # function of W
      error = np.average(loss.sum(axis=1))
      
      if i % 10 == 0:
        print "iteration %d: loss %f" % (i, error)
      
      dscores = self.predict_proba(X)

      dscores[range(self.X.shape[0]),self.Y] -= 1 
      dscores /= X.shape[0]
      
      gradient = np.dot(dscores.T, X) #/ X.shape[0] # 
      db = np.sum(dscores, axis=0, keepdims=True)

      self.W += -learning_rate * gradient
      self.B += -learning_rate * db
      
    
    return self
    
  def predict(self, X):
    #get logit scores
    scores = self.logit(X) + self.B
    
    indices = scores.argmax(axis=1)
    return self.classes[indices]
  
    

import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=4)


print "MINE:"
gd=GradientDescent("")
gd.fit2(X_train, Y_train)
print gd.W
print "Predictions:"
preds =  gd.predict(X_test)
print preds
score = np.average(preds == Y_test)
print score


print "\nSCIKIT: "
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(X_train, Y_train)
print clf.coef_
print "Predictions:"
preds =  clf.predict(X_test)
print preds
score = np.average(preds == Y_test)
print score



        
