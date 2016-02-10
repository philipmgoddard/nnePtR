#' forwardProp performs a forward propogation through the network,
#' using current value of parameters Theta
#'
#' @param unrollThetas vector comprised of unrolled parameter matrix elements
#' @param Thetas template list of matrices allocated to correct size
#' @param nUnits number of units in hidden layers
#' @param nLayers number of hidden layers
#' @param lambda penalty ("weight-decay") term
#' @param outcome matrix of 'dummied' outcomes
#' @param a template list of activation matrices (with "zeroth" layer filled in with inputs)
#' @param z template list of transformed activation matrices
#' @param gradient template list of matrices of gradients
#' @param delta template list of matrices of errors
#' @export
#'
forwardProp <- function(unrollThetas,
                        Thetas, nUnits, nLayers, lambda, outcome,
                        a, z, gradient, delta) {
  # note that gr needs same inputs as fn for use in optim, even if not used
  # gradient, delta and Deltas not needed
  m <- nrow(outcome)
  Thetas <- rollParams_c(unrollThetas, nLayers, Thetas)

  tmp <- propogate_c(Thetas, a, z, nUnits, nLayers)
  a <- tmp[[1]]

  # cost
  J <- (1.0 / m) *
    sum(colSums((-outcome) * log(a[[nLayers + 2]]) -
                  (1.0 - outcome) * log(1.0 - a[[nLayers + 2]]) ))

  # penalty
  penalty <- sum(unlist(lapply(Thetas,
                               function(x) {
                                 (lambda / (2.0 * m)) * sum(colSums(x[, 2:ncol(x), drop = FALSE] ^ 2))
                               }) ))

  J <- J + penalty
  return(J)
}

#' propogate performs the meat of the work for forwardProp
#'
#' @param Thetas list of matrices of parameters
#' @param nUnits number of units in hidden layers
#' @param nLayers number of hidden layers
#' @param a template of activation matrices (with "zeroth" layer filled in with inputs)
#' @param z template of transformed activation matrices
#'
propogate <- function(Thetas, a, z, nUnits, nLayers) {
  for(i in 1:nLayers) {
    z[[i]] <- a[[i]] %*% t(Thetas[[i]])
    a[[i + 1]][, 2:ncol(a[[i + 1]])] <- sigmoid_c(z[[i]])
  }

  z[[nLayers + 1]] <- a[[nLayers + 1]] %*% t(Thetas[[nLayers + 1]])
  a[[nLayers + 2]] <- sigmoid_c(z[[nLayers + 1]])

  return(list(a = a,
              z = z))
}

#' @describeIn propogate
#'
propogate_c <- compiler::cmpfun(propogate)
