#include "../include/mnist_loader.h"
#include <fstream>
#include <iostream>
#include <stdexcept>

MNISTLoader::MNISTLoader(const std::string& dataDir) : dataDir(dataDir) {
    // Add trailing slash if needed
    if (!dataDir.empty() && dataDir.back() != '/') {
        this->dataDir += '/';
    }
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
MNISTLoader::loadTrainingData(int maxImages) {
    std::string imagesPath = dataDir + trainImagesFile;
    std::string labelsPath = dataDir + trainLabelsFile;
    
    auto images = readIDXImages(imagesPath, maxImages);
    auto labels = readIDXLabels(labelsPath, maxImages);
    
    return std::make_tuple(images, labels);
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
MNISTLoader::loadTestData(int maxImages) {
    std::string imagesPath = dataDir + testImagesFile;
    std::string labelsPath = dataDir + testLabelsFile;
    
    auto images = readIDXImages(imagesPath, maxImages);
    auto labels = readIDXLabels(labelsPath, maxImages);
    
    return std::make_tuple(images, labels);
}

std::vector<std::vector<double>> MNISTLoader::readIDXImages(const std::string& filename, int maxImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read header
    // Magic number (should be 2051)
    int magicNumber = readInt(file, 4);
    if (magicNumber != 2051) {
        throw std::runtime_error("Invalid magic number in images file: " + std::to_string(magicNumber));
    }
    
    // Number of images
    int numImages = readInt(file, 4);
    
    // Rows and columns
    int numRows = readInt(file, 4);
    int numCols = readInt(file, 4);
    
    // Limit number of images if requested
    if (maxImages > 0 && maxImages < numImages) {
        numImages = maxImages;
    }
    
    // Read image data
    std::vector<std::vector<double>> images(numImages, std::vector<double>(numRows * numCols));
    
    for (int i = 0; i < numImages; ++i) {
        for (int j = 0; j < numRows * numCols; ++j) {
            // Read pixel value (0-255) and normalize to [0,1]
            unsigned char pixel;
            file.read(reinterpret_cast<char*>(&pixel), 1);
            images[i][j] = static_cast<double>(pixel) / 255.0;
        }
    }
    
    return images;
}

std::vector<std::vector<double>> MNISTLoader::readIDXLabels(const std::string& filename, int maxImages) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read header
    // Magic number (should be 2049)
    int magicNumber = readInt(file, 4);
    if (magicNumber != 2049) {
        throw std::runtime_error("Invalid magic number in labels file: " + std::to_string(magicNumber));
    }
    
    // Number of labels
    int numLabels = readInt(file, 4);
    
    // Limit number of labels if requested
    if (maxImages > 0 && maxImages < numLabels) {
        numLabels = maxImages;
    }
    
    // Read label data
    std::vector<std::vector<double>> labels(numLabels);
    
    for (int i = 0; i < numLabels; ++i) {
        unsigned char label;
        file.read(reinterpret_cast<char*>(&label), 1);
        labels[i] = oneHotEncode(static_cast<int>(label));
    }
    
    return labels;
}

std::vector<double> MNISTLoader::oneHotEncode(int label) {
    if (label < 0 || label > 9) {
        throw std::runtime_error("Invalid label: " + std::to_string(label));
    }
    
    std::vector<double> encoded(10, 0.0);
    encoded[label] = 1.0;
    return encoded;
}

int MNISTLoader::readInt(std::ifstream& file, int count) {
    int value = 0;
    for (int i = 0; i < count; ++i) {
        unsigned char byte;
        file.read(reinterpret_cast<char*>(&byte), 1);
        value = (value << 8) | byte;
    }
    return value;
}