forwardProp <- function(unrollThetas,
                        Thetas, nUnits, nLayers, lambda, outcome,
                        a, z, gradient,
                        delta, Deltas) {
  # number of training samples
  # and number of possible outcomes
  m <- nrow(outcome)
  #nY <- ncol(outcome)

  #roll up thetas back into list of matrices
  Thetas <- rollParams(unrollThetas, nLayers, Thetas)

  # forward propogation
  tmp <- propogate(Thetas, a, z, nUnits, nLayers)
  a <- tmp[[1]]

  # cost
  J <- (1.0 / m) *
    sum(colSums((-outcome) * log(a[[nLayers + 2]]) -
                  (1.0 - outcome) * log(1.0 - a[[nLayers + 2]]) )
    )

  # penalty terms
  penalty <- sum(unlist(lapply(Thetas,
                               function(x) {
                                 (lambda / (2.0 * m)) * sum(colSums(x[, 2:ncol(x), drop = FALSE] ^ 2))
                               })
  ))

  # add penalty to cost
  J <- J + penalty
  return(J)
}



propogate <- function(Thetas, a, z, nUnits, nLayers) {
  for(i in 1:nLayers) {
    z[[i]] <- a[[i]] %*% t(Thetas[[i]])
    a[[i + 1]][, 2:ncol(a[[i + 1]])] <- sigmoid(z[[i]])
  }
  z[[nLayers + 1]] <- a[[nLayers + 1]] %*% t(Thetas[[nLayers + 1]])
  a[[nLayers + 2]] <- sigmoid(z[[nLayers + 1]])

  return(list(a = a,
              z = z))
}
