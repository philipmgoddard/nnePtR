# backprop performs a forward then backward proogation
# to calculatethe gradient of Theta (back prop)
# We want to minimise cost over Thetas eventually
# Arguments - a vector of 'unrolled' Theta matrices,
# nUnits, nLayers, lambda, outcome
# rest are templates of sizes
#
backProp <- function(unrollThetas,
                     Thetas, nUnits, nLayers, lambda, outcome,
                     a, z, gradient,
                     delta, Deltas) {

  m <- nrow(outcome)
  Thetas <- rollParams(unrollThetas, nLayers, Thetas)

  # forward prop
  for(i in 1:nLayers) {
    z[[i]] <- a[[i]] %*% t(Thetas[[i]])
    a[[i + 1]][, 2:ncol(a[[i + 1]])] <- sigmoid(z[[i]])
  }
  z[[nLayers + 1]] <- a[[nLayers + 1]] %*% t(Thetas[[nLayers + 1]])
  a[[nLayers + 2]] <- sigmoid(z[[nLayers + 1]])

  # back propogation to determine errors
  delta[[nLayers + 1]] <- a[[nLayers + 2]] - outcome
  for (i in nLayers:1) {
    delta[[i]] <- (delta[[i + 1]] %*% Thetas[[i + 1]])[, 2:ncol(Thetas[[i + 1]]), drop = FALSE] *
      sigmoidGradient(z[[i]])
  }

  # Deltas and gradient
  for (i in 1:(nLayers + 1)) {
    Deltas[[i]] <- t(delta[[i]]) %*% a[[i]]
    gradient[[i]] <- Deltas[[i]] / m
    gradient[[i]][, 2:ncol(gradient[[i]])] <- gradient[[i]][, 2:ncol(gradient[[i]]), drop = FALSE] +
      (lambda / m) * Thetas[[i]][, 2:ncol(gradient[[i]]), drop = FALSE]
  }

  # unroll gradient so suitable for use with optim
  unrollGrad <- unrollParams(gradient)
  return(unrollGrad)
}
