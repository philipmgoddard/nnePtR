#' @include class-definition.R
NULL

#' Show generic
#'
#' @param object object of class nnePtR
#' @export
#'
setMethod(
  f = "show",
  signature = "nnePtR",
  function(object) {
    cat("\nObject of class nnePtR")
    cat("\n\nNumber of hidden layers: ")
    cat(object@n_layers)
    cat("\nNumber of units per hidden layer: ")
    cat(object@n_units)
    cat("\nPenalty term: ")
    cat(object@penalty)
    cat("\nNumber of training instances: ")
    cat(nrow(object@input))
    cat("\nNumber of input features: ")
    cat(ncol(object@input))
    cat("\nNumber of output classes: ")
    cat(length(levels(object@outcome)))
  }
)

#' Show generic
#'
#' @param object object of class nnePtR
#' @export
#'
setMethod(
  f = "summary",
  signature = "nnePtR",
  function(object) {
    cat("\nObject of class nnePtR")
    cat("\n\nNumber of hidden layers: ")
    cat(object@n_layers)
    cat("\nNumber of units per hidden layer: ")
    cat(object@n_units)
    cat("\nPenalty term: ")
    cat(object@penalty)
    cat("\nNumber of training instances: ")
    cat(nrow(object@input))
    cat("\nNumber of input features: ")
    cat(ncol(object@input))
    cat("\nNumber of output classes: ")
    cat(length(levels(object@outcome)))
    cat("\n\nFinal cost: ")
    cat(object@cost)
    cat("\nOptimisation method: ")
    cat(object@info[[1]])
    cat("\nMax iterations: ")
    cat(object@info[[2]])
    cat("\nConvergence code: ")
    cat(object@info[[3]])
  }
)


#' Predict generic
#'
#' @param object object of class nnePtR
#' @param newdata data to generate predictions for
#' @param type select type = "response" for predicted class, or "prob" for probabilities
#' @export
#'
setMethod(
  f = "predict",
  signature = "nnePtR",
  function(object, newdata, type = "response") {

    # step 1: load params
    newdata <- data.matrix(newdata)
    Thetas <- object@fitted_params
    nSample <- nrow(newdata)
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

    # step 3: return classes or probabilities. if 1 class problem, return
    # probability of positive class, if more that 1 class return p of all classes
    if(dim(a[[length(a)]])[2] == 1) {
      if(type == "response") {
        return(levels(object@outcome)[ifelse(a[[length(a)]] >= 0.5, 1, 0) + 1])
      } else if(type == "prob") {
        tmp <- a[[length(a)]]
        colnames(tmp) <- levels(object@outcome)[1]
        return(a[[length(a)]])
      } else {
        "Please specify type to equal \"response\" or \"prob\""
      }
    } else {
      if(type == "response") {
        return(levels(object@outcome)[max.col(a[[length(a)]])])
      } else if(type == "prob") {
        tmp <- a[[length(a)]]
        colnames(tmp) <- levels(object@outcome)
        return(tmp)
      } else {
        "Please specify type to equal \"response\" or \"prob\""
      }
    }
  }
)
