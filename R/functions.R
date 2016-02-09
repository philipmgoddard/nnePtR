#' Sigmoid function
#'
#' @param x input
#'
sigmoid <- function(x) {
  return(1.0 / (1.0 + exp(-x)))
}


#' @describeIn sigmoid
#'
sigmoid_c <- compiler::cmpfun(sigmoid)


#' Sigmoid gradient function
#'
#' @param x input
#'
sigmoidGradient <- function(x) {
  return(sigmoid(x) * (1.0 - sigmoid(x)))
}


#' @describeIn sigmoidGradient
#'
sigmoidGradient_c <- compiler::cmpfun(sigmoidGradient)

#' Unroll a list of matrices into a single vector
#'
#' @param x input list of matrices
#'
unrollParams <- function(x) {
  return(unlist(lapply(x, c)))
}


#' @describeIn unrollParams
#'
unrollParams_c <- compiler::cmpfun(unrollParams)

#' Roll a vector of parameters into a list of matrices
#' defined by a template
#'
#' @param x input vector to be rolled into list of matrices
#' @param nLayers number of hidden layers in neural network
#' @param Thetas_size template list of matrices for x to be rolled into
#'
rollParams <- function(x, nLayers, Thetas_size) {
  pos <- 1
  for(i in 1:(nLayers + 1)) {
    len <- nrow(Thetas_size[[i]]) * ncol(Thetas_size[[i]])
    Thetas_size[[i]][] <- x[pos:(pos + len - 1)]
    pos <- pos + len
  }
  return(Thetas_size)
}


#' @describeIn rollParams
#'
rollParams_c <- compiler::cmpfun(rollParams)


