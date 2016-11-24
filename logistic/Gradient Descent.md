# Gradient Descent

## Batch GD

Also called Vanilla GD, or the trivial solution. Given a cost/error function, take the gradient (i.e. partial derivative) w.r.t to the parameters (weights) `x` for the entire training set.
$$
w  = w - h *  \nabla_w J(w)
$$
where h is the iteration step size and the gradient is defined as:
$$
\frac{df(w)}{dw} = \frac{f(w+ h) - f(w - h)}{2h} \hspace{0.1in} \text{(use instead)}
$$

## Gradient

### Logit Score

$$
X\cdot W + b
$$

### Softmax (sigmoid)

$$
S(x) = p_x = \frac{e^{x}}{ \sum_j e^{x} }
$$

### Cross Entropy

$$
D(x) = L \cdot -log(p_x)
$$

*where L is the target vector as one-hot encoding.*

### Cross-entropy Log-Loss Cost Function

$$
L_i =D(p_x,L)
$$

### Minimize Cost Function

#### Gradient Calculation


$$
\nabla_w(L_w) = \frac{\partial L_i }{ \partial w_k } = p_k - \mathbb{1}(y_i = k)
$$
*The gradient is very simple: the original class  where the kth element is reduced by 1.*

Back Propagate 
$$
\partial_W = X^T \cdot \nabla_w
$$


Update
$$
w  = w - h *  \partial_W
$$








