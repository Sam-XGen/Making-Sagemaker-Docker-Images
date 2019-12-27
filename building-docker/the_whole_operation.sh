#!/usr/bin/env bash

chmod +x run_operation.sh

./run_operation.sh sagemaker-linear-regression
./run_operation.sh sagemaker-ridge-regression
./run_operation.sh sagemaker-lasso-regression
./run_operation.sh sagemaker-logistic-regression

# naive bayes
./run_operation.sh sagemaker-naive-bayes-multinomial

# support vector machines
./run_operation.sh sagemaker-support-vector-classification
./run_operation.sh sagemaker-support-vector-regression


# stochastic gradient descent
./run_operation.sh sagemaker-stochastic-gradient-descent-classification
./run_operation.sh sagemaker-stochastic-gradient-descent-regression


# k-nearest neighbors
./run_operation.sh sagemaker-k-nearest-neighbors-classification
./run_operation.sh sagemaker-k-nearest-neighbors-regression


# decision tree
./run_operation.sh sagemaker-decision-tree-classification
./run_operation.sh sagemaker-decision-tree-regression


# random forest
./run_operation.sh sagemaker-random-forest-classification
./run_operation.sh sagemaker-random-forest-regression


# gradient boosting
./run_operation.sh sagemaker-gradient-boosting-classification
./run_operation.sh sagemaker-gradient-boosting-regression


# basic neural network
./run_operation.sh sagemaker-basic-neural-network-classification
./run_operation.sh sagemaker-basic-neural-network-regression


