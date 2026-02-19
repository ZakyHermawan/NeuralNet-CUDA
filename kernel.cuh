#include <cuda_runtime.h>

#include <vector>
#include <cstddef>
#include <cstdio>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

void NormalizeCUDA(std::vector<float>&h_data, float mean, float std_dev);

std::vector<float> forwardPipelineWrapper(
    const std::vector<float>&inputData,
    const std::vector<float>&w1, const std::vector<float>&b1,
    const std::vector<float>&w2, const std::vector<float>&b2,
    std::vector<float>&h_hiddenLayer, // Output: Saved for backward pass
    size_t batchSize, size_t inSize, size_t hiddenSize, size_t outSize);

void backpropPipelineWrapper(
    const std::vector<float>&dZ2,          // Gradient from output
    const std::vector<float>&hiddenLayer,  // Activation from hidden
    const std::vector<float>&weights2,     // Weights of layer 2
    const std::vector<float>&batch_X,      // Raw input
    const std::vector<float>&weights1,     // Weights of layer 1
    std::vector<float>&gradWeights2, std::vector<float>&gradBias2,
    std::vector<float>&gradWeights1, std::vector<float>&gradBias1,
    size_t batchSize, size_t inputSize, size_t hiddenSize, size_t outputSize);
