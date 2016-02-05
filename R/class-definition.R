#' S4 class definition for a neural network
#'
#' @slot train_input
#' @slot train_outcome
#' @slot n_layers
#' @slot n_units
#' @slot lambda
#' @slot fitted_params
#' @export
setClass(
  Class = "nnePtR",
  slots = list(
    input = "matrix",
    outcome = "factor",
    n_layers = "numeric",
    n_units = "numeric",
    penalty = "numeric",
    fitted_params = "list")
)

# #' Initialiser for nnePtR objects
# #' @param .Object object of class nnePtR
# #' @export
# setMethod(
#   f = "initialize",
#   signature = "nnePtR",
#   definition = function(.Object) {
#      cat("--- nnePtR: initialiser --- \n")
#     return(.Object)
#   }
# )

#' Constructor for nnePtR
#' @import methods
#' @export
nnetBuild <- function(train_input, train_outcome, nLayers = 1, nUnits = 25, lambda = 0.01) {

  # first step - input into a matrix
  train_input <- data.matrix(train_input)
  # outcome into numbers
  outcome_copy <- train_outcome
  train_outcome <- as.numeric(train_outcome)

  # call setup - this loads lists of matrices of correct size into current environment
  nnetTrainSetup(train_input, train_outcome, nLayers, nUnits, seed = 1234)

  unrollThetas <- unrollParams(Thetas_size)

  # minimise cost over parameters (Thetas) using optim()
  params <- optim(unrollThetas,
                  fn = nnePtR::forwardProp,
                  gr = nnePtR::backProp,
                  method = "L-BFGS-B",
                  Thetas = Thetas_size,
                  nUnits = nUnits,
                  nLayers = nLayers,
                  outcome = outcomeMat,
                  lambda = lambda,
                  a = a_size,
                  z = z_size,
                  gradient = grad_size,
                  delta = delta_size,
                  Deltas = Deltas_size,
                  hessian = FALSE,
                  control = list(maxit = 500))

  # extract final matrices of parameters
  Thetas_final <- rollParams(params$par, nLayers, Thetas_size)

  return(new(Class = "nnePtR",
             input = train_input,
             outcome = outcome_copy,
             n_layers = nLayers,
             n_units = nUnits,
             penalty = lambda,
             fitted_params = Thetas_final)
         )
}
