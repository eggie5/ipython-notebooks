##Linear Model

### Train

Find the weights and bias using the training set 
$$
WX+b
$$


### Gradient Descent

We need a metric to measure the error of our predictions, we can call this our error function: E. Our task now is to minimize our error function, meaning at what values is the error the lowest. If we have a continus function, we can find the the minimum by evaluating the derivative at 0. The derivative at 0 is where they function has a local minimum. For our multi dimensional example, we need to compute the partial derivative at 0. However, this discrete world and not a continuous world, so we have to take the derivative numerically/computationally. A common computational method for evaluating a derivative at 0 is called: Gradient Descent.

IS it possible to build an error function on a classifier class result? How can I measure error? what's the error of expected 'Setosa' vs actual 'versacolor' I think this is why in order to get an error function I have to do the the softmax to get probabilities of each class and then take the distance (cross entropy) to the 1 hot encoded classes.

In a regression you can just take the diff of the actual vs the expected number value. that's my metric. Here my metric is log loss or the cross entropy. If I minimize that, i'll have my weights.

$$
L = 1/N  (\Sigma_i WX_i+b)
$$

The Gradient Descent routine will optimize your error function and return the optimal `W` and `b`.

See more about gradient descent here: http://www.eggie5.com/80-gradient-descent

Also called Batch GD or Vanilla GD

#### Normalization

Normalize input to optimize gradient descent


## Deep Learning

[Background on a neural net]

In order to train our network we need need class probabilities. Meaning for a given example, we don't want to know what class it's in, we want to know the probability that it is in each class. For the following reason, we will modify our Logistic Classifier w/ softmax+cross entropy probabilities to train the network:

To learn the weights all NN use back propagation and optimizers. That implies several things:

1. The cost function must be differentiable.
2. The network needs to know not only the best class at each training point but how far are the other classes from the truth.
3. Weights better be small numbers (if one them "fires up" it dominates the whole network).

###Softmax

Map linear model scores to probabilities 

S(WX+b)

If you increase the size of your outputs, the classifier becomes very confident with predictions. If you reduce the size of your outputs the classifier becomes very unsure. We want to classifier to be not sure of itself in the beginning but to gain confidence as it learns. 

Large magnitudes cause our softmax to become very confident and push the values to 0 or 1.
Smaller magnitudes cause the softmax routine to approach the uniform distributions. since all the scores decrease in magnitude, the resulting softmax probabilities will be closer to each other.

See more about softmax here: http://www.eggie5.com/78-machine-learning-softmax-function


### One-hot encoding of targets

We need a way to represent out labels mathematically . Each label will be represented by a vector of length equal to the number of classes. For example: iris: 3, mnist: 24.
Give each target a one-hot encoding so we can compare the outputs of the softmax using cross entropy calculation. Let's call these labels $L$

### Cross Entropy
We use the cross entropy calculation given below to map our probabilities to a classification target. For example, if we have probabilities [.2, .7, 1], then we would take a cross entropy calculation on the given probability and every target then choose the target w/ the highest entropy, e.g: [0, 1, 0]

$$
D(S,L) = -\Sigma_j L_i\cdot log(S_i)
$$



## Minimize Cross Entropy

We now have a metric which we can use to measure error our model:

$$
D(S(WX+b), L)
$$

Let's formalize it into a loss function. What values of W and b can we choose to minimize D?

$$
L = 1/N *  \Sigma_i D(S(WX_i+b), L_i)
$$

This is the average error across the whole training set. I want to minimize this. How can we minimize a function? Evaluate the derivative at 0. How can we do this in a discrete environment? Gradient Descent.

