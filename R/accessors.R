#' @include class-definition.R
NULL

#' Set the generic for accessor (getter) for fitted coefficients
#' @param object object of class nnePtR
#' @export
#'
setGeneric("getParams",
           function(object){
             standardGeneric("getParams")
           })


#' @describeIn getParams
#' @export
#'
setMethod("getParams",
          signature = "nnePtR",
          function(object){
            return(object@fitted_params)
          })
