import numpy as np
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=4)



K = 3 # number of classes
X = X_train # data matrix (each row = single example)
y = Y_train # class labels
D= X.shape[1]

# initialize parameters randomly
W = 0.01 * np.random.randn(D,K) #4x3
b = np.zeros((1,K))

# some hyperparameters
step_size = 1e-1
reg = 1e-1 # regularization strength

epochs=200

# gradient descent loop
num_examples = X.shape[0]
for i in xrange(epochs):
  
  # evaluate class scores, [N x K]
  scores = np.dot(X, W) + b 
  
  # compute the class probabilities
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
  # compute the loss: average cross-entropy loss and regularization
  correct_probs = probs[range(num_examples),y]
  correct_logprobs = -np.log(correct_probs)
  data_loss = np.sum(correct_logprobs)/num_examples
  
  loss = data_loss 
  if i % 10 == 0:
    print "iteration %d: loss %f" % (i, loss)
  
  # compute the gradient on scores
  dscores = probs
  dscores[range(num_examples),y] -= 1 # for each true prob subtract 1
  dscores /= num_examples
  # print dscores
  
  # backpropate the gradient to the parameters (W,b)
  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)

  # perform a parameter update
  W += -step_size * dW
  b += -step_size * db
  
  
# evaluate training set accuracy
scores = np.dot(X, W) + b
predicted_class = np.argmax(scores, axis=1)
print 'training accuracy: %.2f' % (np.mean(predicted_class == y))

