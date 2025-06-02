#ifndef MNIST_LOADER_H
#define MNIST_LOADER_H

#include <vector>
#include <string>
#include <tuple>


class MNISTLoader {
public:
    /**
     * Constructor
     * 
     * @param dataDir Directory containing the MNIST files
     */
    MNISTLoader(const std::string& dataDir);
    
    /**
     * Load training data (images and labels)
     * 
     * @param maxImages Maximum number of images to load (0 for all)
     * @return Tuple of (images, labels) where:
     *      - images is a vector of image vectors (each image is a flattened vector of pixel values)
     *      - labels is a vector of one-hot encoded labels
     */
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    loadTrainingData(int maxImages = 0);
    
    /**
     * Load test data (images and labels)
     * 
     * @param maxImages Maximum number of images to load (0 for all)
     * @return Tuple of (images, labels) where:
     *      - images is a vector of image vectors (each image is a flattened vector of pixel values)
     *      - labels is a vector of one-hot encoded labels
     */
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    loadTestData(int maxImages = 0);

private:
    std::string dataDir;
    const std::string trainImagesFile = "train-images-idx3-ubyte";
    const std::string trainLabelsFile = "train-labels-idx1-ubyte";
    const std::string testImagesFile = "t10k-images-idx3-ubyte";
    const std::string testLabelsFile = "t10k-labels-idx1-ubyte";
    
    /**
     * Read IDX format image file
     * 
     * @param filename Path to the IDX file
     * @param maxImages Maximum number of images to read (0 for all)
     * @return Vector of images, each image is a vector of pixel values (normalized to [0,1])
     */
    std::vector<std::vector<double>> readIDXImages(const std::string& filename, int maxImages);
    
    /**
     * Read IDX format label file
     * 
     * @param filename Path to the IDX file
     * @param maxImages Maximum number of labels to read (0 for all)
     * @return Vector of one-hot encoded labels (10 dimensions for digits 0-9)
     */
    std::vector<std::vector<double>> readIDXLabels(const std::string& filename, int maxImages);
    
    /**
     * Helper to convert a label (0-9) to one-hot encoded vector
     * 
     * @param label The digit label (0-9)
     * @return One-hot encoded vector (10 dimensions)
     */
    std::vector<double> oneHotEncode(int label);
    
    /**
     * Helper to read big-endian integers from binary file
     * 
     * @param file Input file stream
     * @param count Number of bytes to read (1, 2, or 4)
     * @return Integer value
     */
    int readInt(std::ifstream& file, int count);
};

#endif // MNIST_LOADER_H