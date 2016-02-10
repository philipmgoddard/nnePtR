#' backProp performs a forward propogation with current values
#' of Thetas, then performs a backward propogation to calculate
#' gradient needed for optimisation
#'
#' @param unrollThetas vector comprised of unrolled parameter matrix elements
#' @param Thetas template list of matrices allocated to correct size
#' @param nUnits number of units in each hidden layer of network
#' @param nLayers number of hidden layers in the network
#' @param lambda penalty term
#' @param outcome matrix of 'dummied' outcomes
#' @param a template list of activation matrices (with "zeroth" layer filled in with inputs)
#' @param z template list of transformed activation matrices
#' @param gradient template list of matrices of gradients
#' @param delta template list of matrices of errors
#' @export
#'
backProp <- function(unrollThetas,
                     Thetas, nUnits, nLayers, lambda, outcome,
                     a, z, gradient,
                     delta) {

  m <- nrow(outcome)
  Thetas <- rollParams_c(unrollThetas, nLayers, Thetas)

  # forward propogation
  tmp <- propogate_c(Thetas, a, z, nUnits, nLayers)
  a <- tmp[[1]]
  z <- tmp[[2]]

  # back propogation to determine errors
  delta[[nLayers + 1]] <- a[[nLayers + 2]] - outcome
  for (i in nLayers:1) {
    delta[[i]] <- (delta[[i + 1]] %*% Thetas[[i + 1]])[, 2:ncol(Thetas[[i + 1]]), drop = FALSE] *
      sigmoidGradient_c(z[[i]])
  }

  # Deltas and gradient
  for (i in 1:(nLayers + 1)) {
    #Deltas[[i]] <- t(delta[[i]]) %*% a[[i]]
    gradient[[i]] <- (t(delta[[i]]) %*% a[[i]]) / m #Deltas[[i]] / m
    gradient[[i]][, 2:ncol(gradient[[i]])] <- gradient[[i]][, 2:ncol(gradient[[i]]), drop = FALSE] +
      (lambda / m) * Thetas[[i]][, 2:ncol(gradient[[i]]), drop = FALSE]
  }

  # unroll gradient so suitable for use with optim
  unrollGrad <- unrollParams_c(gradient)
  return(unrollGrad)
}
