# Logistic Regression

## Linear Model

The choice of a linear model assumes tht there is a linear combination of dataset features that woudl properly approximate the output function. If this assumption does not hold, we cannot achieve a small error with a linear model.

Like the Linear Regression, the Logistic regression works by tuning the weights for a linear combination. However, in Logistic Regression, we pass the outputs though a sigmoid function called Softmax which will bound our answers between 0-1, or a probability: 

### Logit Score

$$
X\cdot W + b
$$

### Softmax (sigmoid)

If you increase the size of your outputs, the classifier becomes very confident with predictions. If you reduce the size of your outputs the classifier becomes very unsure. We want to classifier to be not sure of itself in the beginning but to gain confidence as it learns. 

Large magnitudes cause our softmax to become very confident and push the values to 0 or 1.
Smaller magnitudes cause the softmax routine to approach the uniform distributions. since all the scores decrease in magnitude, the resulting softmax probabilities will be closer to each other.


$$
S(x) = p_x = \frac{e^{x}}{ \sum_j e^{x} }
$$
*the output of softmax is class probabilities for each sample. See more about softmax here: http://www.eggie5.com/78-machine-learning-softmax-function*

### Cost Function

We need a metric to measure error between our model and the training data. Our goal will be to minimize the error of that model using Gradient Descent. 

#### Cross-entropy log-loss cost function

We use the cross entropy calculation given below to map our probabilities to a classification target. For example, if we have probabilities [.2, .7, 1], then we would take a cross entropy calculation on the given probability and every target then choose the target w/ the highest entropy, e.g: [0, 1, 0]
$$
L_i = L \cdot -log(p_x)
$$
*where L is the target vector as one-hot encoding and px is the softmax class probabilites.*

#### Label Encoding

Give each target a one-hot encoding so we can compare the outputs of the softmax using cross entropy calculation. Let's call these labels `L`. 

## Batch Gradient Descent

Also called Vanilla GD, or the trivial solution. Given a cost/error function, take the gradient (i.e. partial derivative) w.r.t to the parameters (weights) `x` for the entire training set.
$$
w  = w - h *  \nabla_w J(w)
$$
where h is the iteration step size and the gradient is defined as:
$$
\frac{df(w)}{dw} = \frac{f(w+ h) - f(w - h)}{2h} \hspace{0.1in} 
$$

### Minimize Cost Function

$$
L_i = L \cdot -log(p_x)
$$



We want to minimize error to maximize accuracy of our classifier: this is called optimization. We can minimize the error of our classifier by evaluating the gradient of the cost function at 0. 

#### Gradient Calculation


$$
\nabla_w(L_w) = \frac{\partial L_i }{ \partial w_k } = p_k - L
$$
*The gradient is very simple: the original class where the kth element is reduced by 1.*

Back Propagate 
$$
\partial_W = X^T \cdot \nabla_w
$$


Update
$$
w  = w - h *  \partial_W
$$



We run the Gradient Descent routine above iteratively updating the w vector until a terminate condition is met. 

#### Initialization and Termination 

How do we choose the initial weights and how stop the itterative update routine? 

In general, it is safe to initialize the weighs randomly, so as to avoid getting stuck on a perfectly symmetric hilltop.

Termination is a non-trivial topic in optimization. One simple approach is to use an upper-bound on itterations called `epsilon`.  The problem w/ this approach is that there is no guarntee on the quality of the weights. Another approach is to stop when the gradient is equal or close to 0, by setting some threshold. A combination of the two conditions (setting a large upper bound for teh number of iterations, and a small lower bound for teh size of the gradient) usually works well in practice. However, this still poses the risk of getting stuck in a local minima. A better approach is a combination of the three: maximum number of iterations, marginal error improvement and the size of the error itself. 



## Implementation

### Pseudo Code

```fa
Fixed learning rate gradient desent:
1. Initialize the weights for t=0 to w(0)
2. for t=0, 1,2,... do
3. 	Compute the gradient g=∇L(w(t))
4.	Set the direction to move v=-g
5.	Update teh weights: w(t+1) = w(t) + ηv
6. 	Iterate to the next step until termination condition
7. Return the final weights w.
```



### Python

The following is a  multimodal logistic regression implemention in python following the scikit classifier interface.

```python
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
    
  def get_params(self, deep=True):
    return {}
  
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
    nb_epochs = 500
    params = []
    learning_rate = .1
    thresh = .001
    epsilon = .2 #stop if error is below this
    
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
      #terminate if error is below theshold -- should also check error delta theshold
      if error <= epsilon:
        print "Terminating @ iteration %d: loss %f" % (i, error)
        return

      gradient = (L - self.predict_proba(self.X)).T.dot(self.X)
      
      vector = gradient

      self.W += learning_rate * vector

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
```

#### Evaluation

Let's evaluate our model to the scikit reference implementaiton by comparing 10-fold CV scores:

```
My Implementation:
Accuracy: 0.92 (+/- 0.24)

SGDClassifier: 
Accuracy: 0.78 (+/- 0.24)

LogisticRegression: 
Accuracy: 0.95 (+/- 0.12)
```



Or simple implementation compared well against the reference. We could improve our score by adding in normalization or regularization to help prevent overfitting.