#' forwardPropogate performs a forward propogation through the network,
#' using current value of parameters Theta
#'
#' @param Thetas list of matrices of parameters
#' @param nLayers number of hidden layers
#' @param a template of activation matrices (with "zeroth" layer filled in with inputs)
#'
forwardPropogate <- function(Thetas, a, nLayers) {
  # define size z
  z <- a[2:length(a)]

  for(i in 1:nLayers) {
    z[[i]] <- a[[i]] %*% t(Thetas[[i]])
    a[[i + 1]][, 2:ncol(a[[i + 1]])] <- sigmoid_c(z[[i]])
  }

  z[[nLayers + 1]] <- a[[nLayers + 1]] %*% t(Thetas[[nLayers + 1]])
  a[[nLayers + 2]] <- sigmoid_c(z[[nLayers + 1]])

  return(list(a = a,
              z = z))
}

#' @describeIn forwardPropogate
#'
forwardPropogate_c <- compiler::cmpfun(forwardPropogate)
