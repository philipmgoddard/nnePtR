#' backProp
#'
#' @description Performs a forward propogation with current values
#' of Thetas, then performs a backward propogation to calculate
#' gradient needed for optimisation. The usage is to take advantage of the closure to store
#' templates of matrices, the penalty term, and the outcome.
#' The function then only needs to be passed the unrolled parameters
#' as it is called iteratively for optimisation.
#' @param Thetas template list of matrices allocated to correct size
#' @param lambda penalty term
#' @param outcome matrix of 'dummied' outcomes
#' @param a template list of activation matrices (with "zeroth" layer filled in with inputs)
#'
backProp <- function(Thetas, a, lambda, outcome) {
  # Thetas, a, lambda and outcome stored in enclosing environment

  function(unrollThetas) {
    nLayers <- length(a) - 2
    m <- nrow(outcome)
    Thetas <- rollParams_c(unrollThetas, nLayers, Thetas)

    # forward propogation
    tmp <- forwardPropogate_c(Thetas, a, nLayers)
    a <- tmp[[1]]
    z <- tmp[[2]]

    # cost
    fn <- (1.0 / m) *
      sum(colSums((-outcome) * log(a[[nLayers + 2]]) -
                    (1.0 - outcome) * log(1.0 - a[[nLayers + 2]]) ))

    # penalty
    penalty <- sum(unlist(lapply(Thetas,
                                 function(x) {
                                   (lambda / (2.0 * m)) * sum(colSums(x[, 2:ncol(x), drop = FALSE] ^ 2))
                                 }) ))

    # cost + penalty
    fn <- fn + penalty

    # define sizes of objects base on templates
    delta <- z
    gradient <- Thetas

    # back propogation to determine errors
    delta[[nLayers + 1]] <- a[[nLayers + 2]] - outcome
    for (i in nLayers:1) {
      delta[[i]] <- (delta[[i + 1]] %*% Thetas[[i + 1]])[, 2:ncol(Thetas[[i + 1]]), drop = FALSE] *
        sigmoidGradient_c(z[[i]])
    }

    # gradient
    for (i in 1:(nLayers + 1)) {
      gradient[[i]] <- (t(delta[[i]]) %*% a[[i]]) / m
      gradient[[i]][, 2:ncol(gradient[[i]])] <- gradient[[i]][, 2:ncol(gradient[[i]]), drop = FALSE] +
        (lambda / m) * Thetas[[i]][, 2:ncol(gradient[[i]]), drop = FALSE]
    }
    gr <- unrollParams_c(gradient)

    # return cost and unrolled gradient
    return(list(fnval = fn, grval = gr))
  }

}
