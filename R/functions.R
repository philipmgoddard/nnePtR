#################################################
# sigmoid function

sigmoid <- function(x) {
  return(1.0 / (1.0 + exp(-x)))
}
#################################################


#################################################
# sigmoid gradient function

sigmoidGradient <- function(x) {
  return(sigmoid(x) * (1.0 - sigmoid(x)))
}
#################################################


#################################################
# unroll list of matrices into a vector

unrollParams <- function(x) {
  return(unlist(lapply(x, c)))
}

#################################################
# roll a vector into a list of matrices

rollParams <- function(x, nLayers, Thetas_size) {
  pos <- 1
  for(i in 1:(nLayers + 1)) {
    len <- nrow(Thetas_size[[i]]) * ncol(Thetas_size[[i]])
    Thetas_size[[i]][] <- x[pos:(pos + len - 1)]
    pos <- pos + len
  }
  return(Thetas_size)
}

#################################################
