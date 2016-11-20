import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
import softmax as sm
import gradient_descent as gdr
from sklearn import preprocessing


class LogisticClassifier(object):
  """docstring for LogisticClassifier"""
  
  
  def __init__(self, arg):
    super(LogisticClassifier, self).__init__()
    self.arg = arg
    self.W = [] #matrix of weights
    self.b = 0 #bias
    self.classes = []
    
  def fit(self, X, Y):
    """set the values of W and b WX+b"""
    self.X = X
    self.Y = Y
    
    #get num of classes
    # klasses = np.unique(Y) TODO test this and use: inspiration from scikit
    klasses = list(set(Y))
    self.classes=np.array(klasses)
    
    #randomly initialize weights
    self.W = np.random.randint(100, size=(len(klasses), X.shape[1]))
    
    #Gradient Descent
    gd = gdr.GradientDescent("")
    self.W = gd.fit(X,Y)
  
    print self.X.shape
    
    return self
    
  #decision function
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
    return np.log(self.predict_proba(X)+off)
  
  def log_loss(self, x, y_true):
    #log-loss/cross-entropy
    lb = preprocessing.LabelBinarizer().fit(self.classes)
    
    transformed_labels = lb.transform(y_true)
    y_log_prob = self.predict_log_proba(x)
    loss = -(transformed_labels * y_log_prob)#.sum(axis=1)

    return loss#np.average(loss)
    
    
  def predict(self, X):
    #get logit scores
    scores = self.logit(X)
    
    
    indices = scores.argmax(axis=1)
    return self.classes[indices]
    
    

# TODO In practice, we should scale out data
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X_train)  # Don't cheat - fit only on training data
# X_train = scaler.transform(X_train)
# X_test = scaler.transform(X_test)  # apply same transformation to test data
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=4)

clf = LogisticClassifier("")
print clf.fit(X_train, Y_train)
preds =  clf.predict(X_test)

print "Weights:"
print clf.W

print "Predictions:"
print preds

score = np.average(preds == Y_test)
print score

print clf.predict_proba(X_test)


print "\n\nLOG PROBA ****"
print clf.predict_log_proba(X_test)


print "\n\n Log Loss"
print clf.log_loss(X_train, Y_train)

#then do iternation of gradient
# print clf.log_loss(X_train, Y_train)



