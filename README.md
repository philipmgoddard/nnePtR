## nnePtR

home brewed s4 neural network package

at the moment only suitable for classification

user may define number of hidden layers and number of weights in hidden layers
(must be constant across hidden layers)

solves using optim() with default "L-BFGS-B"

### example usage

```R
library(nnePtR)
model <- nnetBuild(iris[, 1:4], iris[, 5], nLayers = 2, nUnits = 20, lambda = 0.1)

summary(model)
predict(model, newdata = iris[, 1:4], type = "prob")
```

Enjoy!
