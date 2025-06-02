#include "../include/neural_network.h"
#include <iostream>
#include <numeric>
#include <chrono>

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes, double learningRate)
    : layerSizes(layerSizes), learningRate(learningRate) {
    

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng.seed(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    
   
    weights.resize(layerSizes.size() - 1);
    biases.resize(layerSizes.size() - 1);
    
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
  
        double scale = sqrt(2.0 / (layerSizes[i] + layerSizes[i + 1]));
        
        weights[i].resize(layerSizes[i + 1]);
        biases[i].resize(layerSizes[i + 1]);
        
        for (int j = 0; j < layerSizes[i + 1]; ++j) {
            weights[i][j].resize(layerSizes[i]);
            for (int k = 0; k < layerSizes[i]; ++k) {
                weights[i][j][k] = dist(rng) * scale;
            }
            biases[i][j] = dist(rng) * 0.1;
        }
    }
    
    
    activations.resize(layerSizes.size());
    zValues.resize(layerSizes.size() - 1);
    
    for (size_t i = 0; i < layerSizes.size(); ++i) {
        activations[i].resize(layerSizes[i]);
    }
    
    for (size_t i = 0; i < layerSizes.size() - 1; ++i) {
        zValues[i].resize(layerSizes[i + 1]);
    }
}

std::vector<double> NeuralNetwork::feedForward(const std::vector<double>& inputs) {
    if (inputs.size() != static_cast<size_t>(layerSizes[0])) {
        throw std::runtime_error("Input size doesn't match network input layer size");
    }
    
    
    activations[0] = inputs;
    
    
    for (size_t l = 0; l < weights.size(); ++l) {
        for (int j = 0; j < layerSizes[l + 1]; ++j) {
            zValues[l][j] = biases[l][j];
            for (int k = 0; k < layerSizes[l]; ++k) {
                zValues[l][j] += weights[l][j][k] * activations[l][k];
            }
            activations[l + 1][j] = sigmoid(zValues[l][j]);
        }
    }
    
    return activations.back();
}

double NeuralNetwork::train(const std::vector<double>& inputs, const std::vector<double>& targets) {
    if (targets.size() != static_cast<size_t>(layerSizes.back())) {
        throw std::runtime_error("Target size doesn't match network output layer size");
    }
    
   
    feedForward(inputs);
    
   
    std::vector<double> delta(layerSizes.back());
    for (int j = 0; j < layerSizes.back(); ++j) {
        double error = activations.back()[j] - targets[j];
        delta[j] = error * sigmoidPrime(zValues.back()[j]);
    }
    

    std::vector<std::vector<double>> deltas(weights.size());
    deltas.back() = delta;
    
    for (int l = weights.size() - 2; l >= 0; --l) {
        deltas[l].resize(layerSizes[l + 1]);
        
        for (int j = 0; j < layerSizes[l + 1]; ++j) {
            double sum = 0.0;
            for (int k = 0; k < layerSizes[l + 2]; ++k) {
                sum += weights[l + 1][k][j] * deltas[l + 1][k];
            }
            deltas[l][j] = sum * sigmoidPrime(zValues[l][j]);
        }
    }
    
    
    for (size_t l = 0; l < weights.size(); ++l) {
        for (int j = 0; j < layerSizes[l + 1]; ++j) {
            
            biases[l][j] -= learningRate * deltas[l][j];
            
            for (int k = 0; k < layerSizes[l]; ++k) {
                weights[l][j][k] -= learningRate * deltas[l][j] * activations[l][k];
            }
        }
    }
    
   
    double mse = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) {
        double error = activations.back()[i] - targets[i];
        mse += error * error;
    }
    mse /= targets.size();
    
    return mse;
}

double NeuralNetwork::evaluate(const std::vector<std::vector<double>>& testInputs, 
                            const std::vector<std::vector<double>>& testTargets) {
    if (testInputs.size() != testTargets.size()) {
        throw std::runtime_error("Number of test inputs doesn't match number of test targets");
    }
    
    int correctCount = 0;
    
    for (size_t i = 0; i < testInputs.size(); ++i) {
        auto output = feedForward(testInputs[i]);
        

        int predictedDigit = std::distance(output.begin(), 
                                          std::max_element(output.begin(), output.end()));
        int targetDigit = std::distance(testTargets[i].begin(), 
                                       std::max_element(testTargets[i].begin(), testTargets[i].end()));
        
        if (predictedDigit == targetDigit) {
            correctCount++;
        }
    }
    
    return 100.0 * correctCount / testInputs.size();
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double NeuralNetwork::sigmoidPrime(double x) {
    double sigmoidX = sigmoid(x);
    return sigmoidX * (1.0 - sigmoidX);
}