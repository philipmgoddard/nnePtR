#' nnetTrainSetup returns a list of templates for use
#' by forward and back propogation functions.
#' Far more efficient to pass templates (defined once) than
#' redefining matrices of correct size at each iteration when optimising
#'
#' @param input matrix of inputs (nSamples x nFeatures)
#' @param outcome vector of outcomes (factor)
#' @param nLayers number of hidden layers in network
#' @param nUnits number of units in each hidden layer
#' @param seed seed for intilialisng parameters
#'
nnetTrainSetup <- function(input, outcome, nLayers = 1, nUnits = 10, seed = 1234) {
  nFeature <- ncol(input)
  nOutcome <- length(unique(outcome))
  # binary case
  if(nOutcome == 2) nOutcome = 1
  #
  nSample <- nrow(input)

  a_size <- lapply(1:(nLayers + 2), function(x) {
    if(x == 1) {
      tmp <- matrix(NA, nrow = nSample, ncol = nFeature + 1)
      tmp[, 1] <- 1
      tmp[, 2:ncol(tmp)] <- input
      tmp
    }
    else if (x == nLayers + 2) {
      matrix(NA, nrow = nSample, ncol = nOutcome)
    }
    else {
      tmp <- matrix(NA, nrow = nSample, ncol = nUnits + 1)
      tmp[, 1] <- 1
      tmp
    }
  })

  z_size <- lapply(1:(nLayers + 1), function(x) {
    if(x == nLayers + 1) {
      matrix(NA, nrow = nSample, ncol = nOutcome)
    }
    else {
      matrix(NA, nrow = nSample, ncol = nUnits)
    }
  })

  delta_size <- z_size

  set.seed(seed)
  #epsilon_init <- 0.12
  Thetas_size <- lapply(1:(nLayers + 1), function(x) {
    # find more formal way to decide on initial weights
    epsilon_init <- sqrt(6 / dim(a_size[[x]])[1])
    nC <- dim(a_size[[x]])[2]
    # remember bias already included
    # but no bias for output layer (ie s_{j + 1} for last Theta)
    if(x != (nLayers + 1)) {
      nR <- dim(a_size[[x + 1]])[2] - 1
    } else {
      nR <- dim(a_size[[x + 1]])[2]
    }
    matrix(data = (runif(nR * nC) * 2.0 * epsilon_init) - epsilon_init,
           nrow = nR,
           ncol = nC)
  })

  Deltas_size <- lapply(1:(nLayers + 1), function(x) {
    nC <- dim(a_size[[x]])[2]
    if(x != (nLayers + 1)) {
      nR <- dim(a_size[[x + 1]])[2] - 1
    } else {
      nR <- dim(a_size[[x + 1]])[2]
    }
    matrix(data = NA,
           nrow = nR,
           ncol = nC)
  })

  grad_size <- Deltas_size

  # dummy up training outcomes
  outcomeMat <- matrix(data = 0,
                        nrow = length(outcome),
                        ncol = nOutcome)
  for (i in 1:nOutcome) {
    outcomeMat[, i] <- (outcome == i);
  }

  return(list (a_temp = a_size,
               z_temp = z_size,
               delta_temp = delta_size,
               thetas_temp = Thetas_size,
               Deltas_temp = Deltas_size,
               grad_temp = grad_size,
               outcome_temp = outcomeMat))
}
