library(nnePtR)
context("Simple tests")
#
# A basic test for fitting a nnet on iris
#
test_that("calculated cost from test on iris", {
  expect_equal(nnePtR::nnetBuild(iris[, 1:4], iris[, 5], nLayers = 2, nUnits = 20,
                                 lambda = 0.1, seed = 1234, iters = 200)@cost,
               0.1839221,
               tolerance = 1e-6)
})
