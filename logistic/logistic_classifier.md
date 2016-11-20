WX + B = y

X is the input sample, e.g the pixles of letter A or 4 measurments of iris sample
y is a score/prediction, one for each output class: A-Z (26) for letters or 4 for iris
W_k is the matrix of weight vectors for each class

iris example:


vars/features = petal len, petal width, septal len, septal w
weights_virginica = pl_w, pw_w, sl_w, sw_w 
weights_versicacolor = pl_w, pw_w, sl_w, sw_w 
weights_setosa = pl_w, pw_w, sl_w, sw_w 
W=  3x4 matrix of all class weights
W_k = 1x4 weights for a given class
W.T=4x3
bias = b

regression = pl * pl_w + pw * pw_w + sl * sl_w + sw * sw_w + b

X:   1 x 4
W.T: 4 x 3 =
   1 x 3, the logic scores! 
   
   now I can feed that into softmax and get cross etropy score.

find weights and b to minized regression error

## MNIST Example

The weight matrix will have 26 rows (one for each letter/class) and each row will have 768 weights corresponding to each pixel
k=26
X_i = 1 x 768
W_k = 1 x 768 weight vector for a given letter k
W =  26 x 768 matrix of weight vectors for all letters
W.T = 26 x 768

X:    1 x 768
W_k.T: 768 x 26 =
   1 x 26 the logic scores!
   
26 x 1

## one-hot encoding
iris:

01 setosa
10 versacolor
11 virginica

## Cross Entropy 
remapped to column vectors

[[  1.45564071e-022   1.00000000e+000   7.97502652e-120]
 [  1.38316452e-032   9.45878045e-127   5.32162232e-029]
 [  1.00000000e+000   0.00000000e+000   1.00000000e+000]
 ...
 .
 .]

For each element in probabilities above:
  For each class j:
    Run cross entropy

Target 1 setosa
1.45564071e-022         1            = -50
1.38316452e-032         0            = 0
1.00000000e+000         0            = 0

Target 2 versacolor
1.45564071e-022         0            = 0
1.38316452e-032         1            = -73
1.00000000e+000         0            = 0

Target 3 virginica
1.45564071e-022         0            = 0
1.38316452e-032         0            = 0
1.00000000e+000         1            = 0


