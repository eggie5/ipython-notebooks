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


In python:

```python
def evaluate_gradient(loss_function, data, params):
    w=params
    error = loss_function # Nx1
    gradient = np.dot(data.T, error) / N #  4xN- Nx1
    return gradient #4x1
        
data = #the whole training set
params = np.zeros(data.shape[1])
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params -= learning_rate * params_grad
```





