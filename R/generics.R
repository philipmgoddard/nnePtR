#' @include class-definition.R
NULL

#' Show generic
#' @param object object of class nnePtR
#' @export
setMethod(
  f = "show",
  signature = "nnePtR",
  function(object) {
    cat("This neural net has units: ")
    cat(object@n_units)
    cat("\n\nThis neural net has hidden layers: ")
    cat(object@n_layers)
  }
)

#' Predict generic
#' @param object object of class nnePtR
#' @export
setMethod(
  f = "predict",
  signature = "nnePtR",
  function(object, newdata, ...) {

    # step 1: load params
    newdata <- data.matrix(new_data)
    Thetas <- object@fitted_params
    nSample <- nrow(new_data)
    nUnits <- object@n_units
    nLayers <- object@n_layers
    nOutcome <- length(unique(object@outcome))
    nFeature <- ncol(newdata)
    # binary case
    if(nOutcome == 2) nOutcome = 1

    # step 2: define a and z
    a <- lapply(1:(nLayers + 2), function(x) {
      if(x == 1) {
        tmp <- matrix(NA, nrow = nSample, ncol = nFeature + 1)
        tmp[, 1] <- 1
        tmp[, 2:ncol(tmp)] <- newdata
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

    z <- lapply(1:(nLayers + 1), function(x) {
      if(x == nLayers + 1) {
        matrix(NA, nrow = nSample, ncol = nOutcome)
      }
      else {
        matrix(NA, nrow = nSample, ncol = nUnits)
      }
    })

    tmp <- propogate(Thetas, a, z, nUnits, nLayers)
    a <- tmp[[1]]

    # return classes or probabilities. if 1 class problem, return
    # probability of positive class, if more that 1 class return p of all classes
    # use 'type' argument?
    if(dim(a[[length(a)]])[2] == 1) {
      return(list(prob = a[[length(a)]],
                  class = ifelse(a[[length(a)]] >= 0.5, 1, 0)))
    } else {
      return(list(prob = a[[length(a)]],
                  class = max.col(a[[length(a)]])))
    }

  }
)
