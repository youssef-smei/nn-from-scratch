#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <functional>

/**
 * A simple feed-forward neural network implementation.
 * This network can be configured with arbitrary layer sizes and uses
 * sigmoid activation function with the mean squared error loss function.
 */
class NeuralNetwork {
public:
    /**
     * Constructor for the neural network
     * 
     * @param layerSizes Vector containing the number of neurons in each layer
     *                  (including input and output layers)
     * @param learningRate Learning rate for gradient descent
     */
    NeuralNetwork(const std::vector<int>& layerSizes, double learningRate = 0.01);
    
    /**
     * Feed inputs forward through the network
     * 
     * @param inputs Vector of input values
     * @return Vector of output values
     */
    std::vector<double> feedForward(const std::vector<double>& inputs);
    
    /**
     * Train the network using backpropagation
     * 
     * @param inputs Vector of input values
     * @param targets Vector of target output values
     * @return Error (mean squared error)
     */
    double train(const std::vector<double>& inputs, const std::vector<double>& targets);
    
    /**
     * Evaluate the network on test data
     * 
     * @param testInputs Vector of test input sets
     * @param testTargets Vector of test target sets
     * @return Accuracy as a percentage
     */
    double evaluate(const std::vector<std::vector<double>>& testInputs, 
                  const std::vector<std::vector<double>>& testTargets);

private:
    // Network structure
    std::vector<int> layerSizes;
    std::vector<std::vector<std::vector<double>>> weights; // [layer][neuron][prev_neuron]
    std::vector<std::vector<double>> biases; // [layer][neuron]
    
    // Network parameters
    double learningRate;
    
    // For storing intermediate values during forward/backward passes
    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> zValues; 
    
    // Activation function and its derivative
    double sigmoid(double x);
    double sigmoidPrime(double x);
    
    // random number generator
    std::mt19937 rng;
};

#endif 