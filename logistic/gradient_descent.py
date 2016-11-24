from sklearn.linear_model import SGDClassifier
import softmax as sm
from sklearn import preprocessing

class GradientDescent(object):
  """GradientDescent"""
  def __init__(self):
    super(GradientDescent, self).__init__()
    self.W = np.array([]) #matrix of weights
    self.b = 0 #bias
    self.classes = []
  
  def logit(self, X):
    scores = X.dot(self.W.T) + self.B
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
    
  def fit(self, X, Y):
    nb_epochs = 200
    params = []
    learning_rate = .1
    thresh = .001
    
    self.X = X
    self.Y = Y
    
    #get num of classes
    self.classes = np.unique(Y)
    
    #random init weights
    self.W = 0.01 * np.random.randn(len(self.classes),X.shape[1]) #4x3
    self.B = np.zeros((1,len(self.classes)))
    L = lb = preprocessing.LabelBinarizer().fit(self.classes).transform(self.Y)
    
    for i in range(nb_epochs):
      loss = self.loss_function(X, Y) # function of W
      error = np.average(loss.sum(axis=1))
      
      # if i % 10 == 0:
      #   print "iteration %d: loss %f" % (i, error)

      gradient = (L - self.predict_proba(self.X)).T.dot(self.X)
      
      vector = gradient

      self.W += learning_rate * vector
      
      #TODO add stopping conditoin
      
    
    return self
    
  def predict(self, X):
    #get logit scores
    scores = self.logit(X) 
    
    indices = scores.argmax(axis=1)
    return self.classes[indices]
  
  def score(self, X_test, Y_test):
    preds =  self.predict(X_test)
    score = np.average(preds == Y_test)
    return score
    
  def get_params(self, deep=True):
    return {}
    
  
    

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=4)


print "MINE:"
gd=GradientDescent()
gd.fit(X_train, Y_train)
scores = cross_val_score(gd, iris.data, iris.target, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


print "\nSCIKIT: "
clf = SGDClassifier(loss="log", penalty="l2")
clf.fit(X_train, Y_train)
scores = cross_val_score(clf, iris.data, iris.target, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



        
