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
  return(unlist(lapply(x, function(x){
    dim(x) <- nrow(x) * ncol(x)
    x})
    ))
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


#' Function operator to catch exceptions for optimisation when fitting neural network.
#' Courtesy Hadley Wickam.
#'
#' @param default default value to return when exception occurs
#' @param f function to try
#' @param quiet logical. Should the functin fail silently?
#'
failwith <- function(default = NULL, f, quiet = TRUE) {
  force(f)
  function(...) {
    out <- default
    try(out <- f(...), silent = quiet)
    out
  }
}

#' Split function. As gradient and cost both calculated in backProp function
#' we want to cache values as optim will call the functions seperately (for fn and gr).
#' This closure splits the function, and will check if x (the par argument of optim)
#' changes between calls. Will retrieve cached values if not (i.e will calculate cost
#' and gradient for for fn, and retrieve cached values for gr)
#'
#' @param f the function to be split
#'
splitfn <- function(f) {
  lastx <- NA
  lastfn <- NA
  lastgr <- NA

  doeval <- function(x, ...) {
    if (identical(all.equal(x, lastx), TRUE)) return(lastfn)
    lastx <<- x
    both <- f(x, ...)
    lastfn <<- both$fnval
    lastgr <<- both$grval
    return(lastfn)
  }

  fn <- function(x, ...) doeval(x, ...)

  gr <- function(x, ...) {
    doeval(x, ...)
    lastgr
  }

  return(list(fn = fn,
              gr = gr))
}
