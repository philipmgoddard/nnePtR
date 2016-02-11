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


#' split function - make a closure to speed up use of optim by caching. now we just need backprop!
#' modify backprop so it calculates the cost.
#' Then first pass of backprop- calcs cost and gradient + caches. second call, retrieves cached gradient
#' @param f the function to be split
#'
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

  list(fn=fn, gr=gr)
}

# fr <- function(x, b) {   ## Rosenbrock Banana function - gradient and function both returned
#   Sys.sleep(0.001)
#   x1 <- x[1]
#   x2 <- x[2]
#   fn <- b * 100 * (x2 - x1 * x1)^2 + (1 - x1)^2
#   gr <- c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1), 200*(x2 - x1 * x1))
#   return(list(fnval = fn, grval = gr))
# }
#
# f2 <- splitfn(fr)
# f2$fn(x = c(-1, -1), b = 3)
# #
# #
# # grr <- function(x) { ## Gradient of 'fr'
# #   Sys.sleep(0.001)
# #
# #   x1 <- x[1]
# #   x2 <- x[2]
# #   c(-400 * x1 * (x2 - x1 * x1) - 2 * (1 - x1),
# #     200 *      (x2 - x1 * x1))
# # }
# # frr <- function(x) {   ## Rosenbrock Banana function
# #   Sys.sleep(0.001)
# #
# #   x1 <- x[1]
# #   x2 <- x[2]
# #   100 * (x2 - x1 * x1)^2 + (1 - x1)^2
# # }
# #
# #
# optim(c(-1.2,1), fn = f2$fn, gr =f2$gr, b = 3, method = "BFGS")
# #
# # optim(c(-1.2,1), frr, grr, method = "BFGS")
