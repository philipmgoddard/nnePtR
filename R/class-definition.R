#' S4 class definition for a neural network
#'
#' @slot train_input
#' @slot train_outcome
#' @slot levels
#' @slot n_layers
#' @slot n_units
#' @slot penalty
#' @slot cost
#' @slot lambda
#' @slot fitted_params
#' @slot info
#'
setClass(
  Class = "nnePtR",
  slots = list(
    input = "matrix",
    outcome = "factor",
    levels = "character",
    n_layers = "numeric",
    n_units = "numeric",
    penalty = "numeric",
    cost = "numeric",
    fitted_params = "list",
    info = "list")
)

#' Constructor for nnePtR
#'
#' @param train_input data frame or matrix of input features
#' @param train_outcome vector of outcome. should be a factor
#' @param nLayers number of hidden layers
#' @param nUnits number of units in hidden layers. Constant across each hidden layer.
#' @param lambda penalty term
#' @param seed seed for initialising network
#' @param iters number of iterations (passed to optim)
#' @param optim_method opitimisation method (passed to optim)
#' @import methods
#' @export
#'
nnetBuild <- function(train_input, train_outcome, nLayers = 1, nUnits = 25,
                      lambda = 0.01, seed = 1234,
                      iters = 200, optim_method = "L-BFGS-B") {

  train_input <- data.matrix(train_input)
  outcome_copy <- train_outcome
  train_outcome <- as.numeric(train_outcome)

  seed_tmp <- seed
  repeat {

    templates <- nnetTrainSetup_c(train_input,
                                  train_outcome,
                                  nLayers,
                                  nUnits,
                                  seed = seed_tmp)
    a_size <- templates[[1]]
    z_size <- templates[[2]]
    delta_size <- templates[[3]]
    Thetas_size <- templates[[4]]
    Deltas_size <- templates[[5]]
    grad_size <- templates[[6]]
    outcomeMat <- templates[[7]]

    unrollThetas <- unrollParams_c(Thetas_size)


    params <- failwith(NULL, optim)(unrollThetas,
                                   fn = nnePtR::forwardProp,
                                   gr = nnePtR::backProp,
                                   method = "L-BFGS-B",
                                   Thetas = Thetas_size,
                                   nUnits = nUnits,
                                   nLayers = nLayers,
                                   lambda = lambda,
                                   outcome = outcomeMat,
                                   a = a_size,
                                   z = z_size,
                                   gradient = grad_size,
                                   delta = delta_size,
                                   Deltas = Deltas_size,
                                   hessian = FALSE,
                                   control = list(maxit = iters))

    if(!is.null(params)) break
    seed_tmp <- seed_tmp + 1
  }

  Thetas_final <- rollParams_c(params$par, nLayers, Thetas_size)

  return(new(Class = "nnePtR",
             input = train_input,
             outcome = outcome_copy,
             levels = levels(outcome_copy),
             n_layers = nLayers,
             n_units = nUnits,
             penalty = lambda,
             cost = params$value,
             fitted_params = Thetas_final,
             info = list(method = optim_method,
                         max_iterations = iters,
                         convergence = params$convergence))
  )

}

#' Initializor to catch input errors
#'
#' @param .Object object of class nnePtR
#' @param input matrix of inputs. Dimensions are (n training samples x n features)
#' @param outcome factor variable for outcome
#' @param levels character vector of levels of outcome
#' @param n_layers number of hidden layers in the network
#' @param n_units number of units in the hidden layers
#' @param penalty penalty term (weight decay)
#' @param cost final cost obtained through optimisation
#' @param fitted_params list of matrices holding the fitted parameters
#' @param info list containing infomation from optimisation
#'
setMethod(
  f="initialize",
  signature = "nnePtR",
  definition = function(.Object, input, outcome, levels,
                        n_layers, n_units, penalty,
                        cost, fitted_params, info) {

    if(nrow(input) != length(outcome)) stop("input and outcome do not match")
    if(n_layers < 1) stop("there must be at least 1 hidden layer")
    if(!class(outcome) == "factor") stop("outcome must be a factor variable")
    if(!class(input) == "matrix") stop("input must be a matrix variable")
    if(sum(is.na(input)) >= 1) stop("missing values in input")
    if(info$convergence > 1) print("it is advised to investigate convergence of optimisation")

    .Object@input <- input
    .Object@outcome <- outcome
    .Object@levels <- levels
    .Object@n_layers <- n_layers
    .Object@n_units <- n_units
    .Object@penalty <- penalty
    .Object@cost <- cost
    .Object@fitted_params <- fitted_params
    .Object@info <- info
    return(.Object)
  }
)
