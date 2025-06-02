#include "../include/neural_network.h"
#include "../include/mnist_loader.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <numeric>

int main(int argc, char* argv[]) {
    try {
        std::string dataDir = "./data";
        if (argc > 1) {
            dataDir = argv[1];
        }
        
        std::cout << "Loading MNIST data from: " << dataDir << std::endl;
        
        
        MNISTLoader mnistLoader(dataDir);
        
        
        const int trainingSize = 10000;
        std::cout << "Loading " << trainingSize << " training examples..." << std::endl;
        auto [trainingImages, trainingLabels] = mnistLoader.loadTrainingData(trainingSize);
        std::cout << "Loaded " << trainingImages.size() << " training examples" << std::endl;
        
        const int testSize = 1000;
        std::cout << "Loading " << testSize << " test examples..." << std::endl;
        auto [testImages, testLabels] = mnistLoader.loadTestData(testSize);
        std::cout << "Loaded " << testImages.size() << " test examples" << std::endl;
        
        std::vector<int> layerSizes = {784, 30, 10};
        double learningRate = 0.1;
        
        std::cout << "Creating neural network with layers: ";
        for (size_t i = 0; i < layerSizes.size(); ++i) {
            std::cout << layerSizes[i];
            if (i < layerSizes.size() - 1) std::cout << " -> ";
        }
        std::cout << std::endl;
        
        NeuralNetwork network(layerSizes, learningRate);
        
       
        const int epochs = 10;
        const int batchSize = 100;
        

        std::vector<int> indices(trainingImages.size());
        std::iota(indices.begin(), indices.end(), 0);
        
        
        std::cout << "\nTraining for " << epochs << " epochs with batch size " << batchSize << "..." << std::endl;
        auto startTime = std::chrono::high_resolution_clock::now();
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            
            std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
            
            double totalError = 0.0;
            int numBatches = trainingImages.size() / batchSize;
            
            for (int batch = 0; batch < numBatches; ++batch) {
                double batchError = 0.0;
                
                for (int i = 0; i < batchSize; ++i) {
                    int idx = indices[batch * batchSize + i];
                    batchError += network.train(trainingImages[idx], trainingLabels[idx]);
                }
                
                totalError += batchError / batchSize;
                
                
                if ((batch + 1) % 10 == 0) { 
                    std::cout << "Epoch " << epoch + 1 << ", Batch " << batch + 1 << "/" << numBatches
                              << ", Error: " << std::fixed << std::setprecision(6) << (batchError / batchSize)
                              << "\r" << std::flush;
                }
            }
            
            
            double accuracy = network.evaluate(testImages, testLabels);
            
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " completed, Avg Error: " << std::fixed << std::setprecision(6) << (totalError / numBatches)
                      << ", Test Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        }
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        
        std::cout << "\nTraining completed in " << duration << " seconds" << std::endl;
        
        
        double finalAccuracy = network.evaluate(testImages, testLabels);
        std::cout << "Final test accuracy: " << std::fixed << std::setprecision(2) << finalAccuracy << "%" << std::endl;
        
        
        std::cout << "\nShowing predictions for 5 test examples:" << std::endl;
        for (int i = 0; i < 1000; ++i) {
            auto output = network.feedForward(testImages[i]);
            int predictedDigit = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            int actualDigit = std::distance(testLabels[i].begin(), std::max_element(testLabels[i].begin(), testLabels[i].end()));
            
            std::cout << "Example " << i + 1 << ": Predicted " << predictedDigit << ", Actual " << actualDigit << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}