% Generated by roxygen2 (4.1.1): do not edit by hand
% Please edit documentation in R/setup.R
\name{nnetTrainSetup}
\alias{nnetTrainSetup}
\alias{nnetTrainSetup_c}
\title{nnetTrainSetup returns a list of templates for use
by forward and back propogation functions.
Far more efficient to pass templates (defined once) than
redefining matrices of correct size at each iteration when optimising}
\usage{
nnetTrainSetup(input, outcome, nLayers = 1, nUnits = 10, seed = 1234)

nnetTrainSetup_c(input, outcome, nLayers = 1, nUnits = 10, seed = 1234)
}
\arguments{
\item{input}{matrix of inputs (nSamples x nFeatures)}

\item{outcome}{vector of outcomes (factor)}

\item{nLayers}{number of hidden layers in network}

\item{nUnits}{number of units in each hidden layer}

\item{seed}{seed for intilialisng parameters}
}
\description{
nnetTrainSetup returns a list of templates for use
by forward and back propogation functions.
Far more efficient to pass templates (defined once) than
redefining matrices of correct size at each iteration when optimising
}
\section{Functions}{
\itemize{
\item \code{nnetTrainSetup_c}: 
}}

