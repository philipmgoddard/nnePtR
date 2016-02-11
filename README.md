## nnePtR

home brewed s4 neural network package

at the moment only suitable for classification

user may define number of hidden layers and number of weights in hidden layers
(must be constant across hidden layers)

solves using optim() with default "L-BFGS-B"

##TODO:
redo backprop as a closure to store theta size and a size templates. Should get some
speedup there as wont have to pass these through optim as every iteration
