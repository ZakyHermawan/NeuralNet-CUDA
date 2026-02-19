
#include "kernel.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

#define BLOCK_SIZE 16

/**
 * @brief CUDA kernel for element-wise data normalization.
 * Performs: data[i] = (data[i] - mean) / std_dev
 * * @param data     Pointer to the global memory array to be modified in-place.
 * @param mean     The pre-calculated average value of the dataset.
 * @param inv_std  The pre-calculated inverse of the standard deviation (1.0 / std_dev).
 * We pass the inverse to replace a slow division with a fast multiplication.
 * @param size     The total number of elements in the array to prevent out-of-bounds access.
 */
static __global__ void NormalizeKernel(float* data, float mean, float inv_std, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        data[idx] = (data[idx] - mean) * inv_std;
    }
}

void NormalizeCUDA(std::vector<float>& h_data, float mean, float std_dev) {
    size_t size = h_data.size();
    size_t bytes = size * sizeof(float);
    float inv_std = (std_dev > 0.0f) ? (1.0f / std_dev) : 1.0f;

    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    NormalizeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, (float)mean, inv_std, size);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_data));
}

/**
 * @brief Fused Kernel for Linear Forward Pass with Optional ReLU.
 * Performs: Output = σ(Input * Weights + Bias)
 * * This kernel uses Shared Memory Tiling to minimize Global Memory bandwidth
 * bottlenecks (The "Memory Wall").
 *
 * @param input      [A x B] Matrix (e.g., Batch Size x Input Features)
 * @param weights    [B x C] Matrix (e.g., Input Features x Output Neurons)
 * @param bias       [C] Vector     (e.g., Output Neurons)
 * @param output     [A x C] Matrix (e.g., Batch Size x Output Neurons)
 * @param A, B, C    Matrix dimensions for MatMul (A x B) * (B x C)
 * @param applyReLU  Flag to fuse ReLU activation into this pass
 */
static __global__ void FusedLinearReLUForwardKernel(
    const float* input, const float* weights, const float* bias,
    float* output, int A, int B, int C, bool applyReLU)
{
    __shared__ float s_input[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_weights[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (B + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        if (row < A && t * BLOCK_SIZE + threadIdx.x < B)
        {
            s_input[threadIdx.y][threadIdx.x] = input[row * B + t * BLOCK_SIZE + threadIdx.x];
        }
        else
        {
            s_input[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < C && t * BLOCK_SIZE + threadIdx.y < B)
        {
            s_weights[threadIdx.y][threadIdx.x] = weights[(t * BLOCK_SIZE + threadIdx.y) * C + col];
        }
        else
        {
            s_weights[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += s_input[threadIdx.y][k] * s_weights[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < A && col < C)
    {
        float val = sum + bias[col];
        if (applyReLU && val < 0.0f) val = 0.0f;
        output[row * C + col] = val;
    }
}

std::vector<float> forwardPipelineWrapper(
    const std::vector<float>& inputData,
    const std::vector<float>& w1, const std::vector<float>& b1,
    const std::vector<float>& w2, const std::vector<float>& b2,
    std::vector<float>& h_hiddenLayer,
    size_t batchSize, size_t inSize, size_t hiddenSize, size_t outSize)
{
    float* d_in, * d_w1, * d_b1, * d_hidden, * d_w2, * d_b2, * d_out;

    CUDA_CHECK(cudaMalloc(&d_in, batchSize * inSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, inSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1, hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, hiddenSize * outSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2, outSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, batchSize * outSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, inputData.data(), batchSize * inSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, w1.data(), inSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b1, b1.data(), hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, w2.data(), hiddenSize * outSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b2, b2.data(), outSize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid1((hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Layer 1: Linear + ReLU (Fused)
    FusedLinearReLUForwardKernel << <grid1, block >> > (d_in, d_w1, d_b1, d_hidden, batchSize, inSize, hiddenSize, true);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Layer 2: Linear Only (No ReLU for final digits/probabilities)
    dim3 grid2((outSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    FusedLinearReLUForwardKernel << <grid2, block >> > (d_hidden, d_w2, d_b2, d_out, batchSize, hiddenSize, outSize, false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> output(batchSize * outSize);

    CUDA_CHECK(cudaMemcpy(output.data(), d_out, batchSize * outSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_hiddenLayer.data(), d_hidden, batchSize * hiddenSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_b1));
    CUDA_CHECK(cudaFree(d_hidden));
    CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_b2));
    CUDA_CHECK(cudaFree(d_out));

    return output;
}

/**
 * @brief Tiled Kernel for Weight and Bias Gradients (dW = X^T * dZ)
 * We tile across the Batch dimension to compute the dot product.
 */
static __global__ void TiledWeightBackwardKernel(
    const float* gradOutput, const float* layerInput,
    float* gradWeights, float* gradBias,
    int batchSize, int inSize, int outSize)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_X[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_dY[BLOCK_SIZE][BLOCK_SIZE];

    float dW_sum = 0.0f;

    // Tile across the Batch dimension (which is the inner dimension for X^T * dZ)
    for (int t = 0; t < (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {

        // Load X^T tile into shared memory (Transposing on the fly)
        int x_r = t * BLOCK_SIZE + threadIdx.x; // Batch index
        int x_c = row;                          // inSize index
        s_X[threadIdx.y][threadIdx.x] = (x_r < batchSize && x_c < inSize) ? layerInput[x_r * inSize + x_c] : 0.0f;

        // Load dY tile into shared memory
        int dy_r = t * BLOCK_SIZE + threadIdx.y; // Batch index
        int dy_c = col;                          // outSize index
        s_dY[threadIdx.y][threadIdx.x] = (dy_r < batchSize && dy_c < outSize) ? gradOutput[dy_r * outSize + dy_c] : 0.0f;

        __syncthreads();

        // Compute dot product for this specific [BLOCK x BLOCK] tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            dW_sum += s_X[threadIdx.y][k] * s_dY[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write final dW. We use '=' because we computed the entire sum. No atomics needed.
    if (row < inSize && col < outSize) {
        gradWeights[row * outSize + col] = dW_sum;
    }

    // Compute Bias gradient (db = sum(dY across batch))
    if (row == 0 && col < outSize) {
        float db_sum = 0.0f;
        for (int b = 0; b < batchSize; ++b) {
            db_sum += gradOutput[b * outSize + col]; // Reads down the batch column
        }
        gradBias[col] = db_sum;
    }
}

/**
 * @brief Tiled Kernel for Input Gradients (dX = dZ * W^T)
 * We tile across the Output Feature dimension to compute the dot product.
 */
static __global__ void TiledInputBackwardKernel(
    const float* gradOutput, const float* weights, const float* currentLayerZ,
    float* gradInput, int batchSize, int inSize, int outSize)
{
    // Thread geometry maps to the Output Input Matrix (dX): [batchSize x inSize]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float s_dY[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_W[BLOCK_SIZE][BLOCK_SIZE];

    float dX_sum = 0.0f;

    // Tile across the outSize dimension
    for (int t = 0; t < (outSize + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {

        // Load dY tile
        int dy_r = row;
        int dy_c = t * BLOCK_SIZE + threadIdx.x;
        s_dY[threadIdx.y][threadIdx.x] = (dy_r < batchSize && dy_c < outSize) ? gradOutput[dy_r * outSize + dy_c] : 0.0f;

        // Load W^T tile (Transposing on the fly)
        int w_r = t * BLOCK_SIZE + threadIdx.y;
        int w_c = col;
        s_W[threadIdx.y][threadIdx.x] = (w_r < outSize && w_c < inSize) ? weights[w_c * outSize + w_r] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            dX_sum += s_dY[threadIdx.y][k] * s_W[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Apply ReLU Derivative and Write final dX
    if (row < batchSize && col < inSize) {
        // Since currentLayerZ holds post-ReLU activations, any value > 0 means the pre-activation was > 0.
        // This makes it a perfect, mathematically sound mask.
        if (currentLayerZ != nullptr) {
            dX_sum *= (currentLayerZ[row * inSize + col] > 0.0f) ? 1.0f : 0.0f;
        }
        gradInput[row * inSize + col] = dX_sum; // Overwrite memory directly
    }
}

void backpropPipelineWrapper(
    const std::vector<float>& dZ2,
    const std::vector<float>& hiddenLayer,
    const std::vector<float>& weights2,
    const std::vector<float>& batch_X,
    const std::vector<float>& weights1,
    std::vector<float>& gradWeights2, std::vector<float>& gradBias2,
    std::vector<float>& gradWeights1, std::vector<float>& gradBias1,
    size_t batchSize, size_t inputSize, size_t hiddenSize, size_t outputSize)
{
    float* d_dZ2, * d_hidden, * d_w2, * d_X, * d_w1;
    float* d_dW2, * d_db2, * d_dW1, * d_db1, * d_dHidden, * d_dummy;

    CUDA_CHECK(cudaMalloc(&d_dZ2, batchSize * outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hidden, batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w2, hiddenSize * outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_X, batchSize * inputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_w1, inputSize * hiddenSize * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dW2, hiddenSize * outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db2, outputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dW1, inputSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_db1, hiddenSize * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_dHidden, batchSize * hiddenSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dummy, batchSize * inputSize * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_dZ2, dZ2.data(), batchSize * outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_hidden, hiddenLayer.data(), batchSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w2, weights2.data(), hiddenSize * outputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, batch_X.data(), batchSize * inputSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w1, weights1.data(), inputSize * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    /* ========================================================================
     * LAYER 2 BACKWARD (Output Layer Gradients)
     * ------------------------------------------------------------------------
     * Variables:
     * A[1] = hiddenLayer (Activations from Layer 1)
     * dZ[2] = d_dZ2 (Error from Loss function, pre-scaled by 1/m on CPU)
     * W[2] = d_w2 (Weights of Layer 2)
     * * Equations:
     * 1. dW[2] = (A[1])^T * dZ[2]
     * 2. db[2] = sum(dZ[2], axis=batch)
     * ======================================================================== */
    dim3 gridW2((outputSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    TiledWeightBackwardKernel << <gridW2, block >> > (d_dZ2, d_hidden, d_dW2, d_db2, (int)batchSize, (int)hiddenSize, (int)outputSize);
    CUDA_CHECK(cudaGetLastError());

    /* ========================================================================
     * ERROR PROPAGATION & RELU DERIVATIVE (Hidden Layer Error)
     * ------------------------------------------------------------------------
     * We need to push the error backwards through Layer 2's weights, and then
     * apply the derivative of the ReLU function used in Layer 1.
     * * Equations:
     * 1. dA[1] = dZ[2] * (W[2])^T                 <- Pushing error through weights
     * 2. dZ[1] = dA[1] .* ReLU_Derivative(Z[1])   <- Hadamard product (element-wise)
     * * Note on ReLU Derivative:
     * Since we cached the post-ReLU activations in 'hiddenLayer', any value > 0
     * means the gate was open. So: ReLU_Deriv = (hiddenLayer > 0) ? 1.0 : 0.0
     * * Combined Kernel Equation:
     * d_dHidden = (d_dZ2 * d_w2^T) .* (d_hidden > 0)
     * ======================================================================== */
    dim3 gridX2((hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    TiledInputBackwardKernel << <gridX2, block >> > (d_dZ2, d_w2, d_hidden, d_dHidden, (int)batchSize, (int)hiddenSize, (int)outputSize);
    CUDA_CHECK(cudaGetLastError());

    /* ========================================================================
     * LAYER 1 BACKWARD (Hidden Layer Gradients)
     * ------------------------------------------------------------------------
     * Variables:
     * X = d_X (Raw Image Input pixels)
     * dZ[1] = d_dHidden (The error we just calculated above)
     * * Equations:
     * 1. dW[1] = (X)^T * dZ[1]
     * 2. db[1] = sum(dZ[1], axis=batch)
     * ======================================================================== */
    dim3 gridW1((hiddenSize + BLOCK_SIZE - 1) / BLOCK_SIZE, (inputSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
    TiledWeightBackwardKernel << <gridW1, block >> > (d_dHidden, d_X, d_dW1, d_db1, (int)batchSize, (int)inputSize, (int)hiddenSize);
    CUDA_CHECK(cudaGetLastError());

    // The cudaMemcpy command acts as our barrier. The CPU will automatically sleep here until 
    // all three kernels above finish executing.
    CUDA_CHECK(cudaMemcpy(gradWeights2.data(), d_dW2, hiddenSize * outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gradBias2.data(), d_db2, outputSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gradWeights1.data(), d_dW1, inputSize * hiddenSize * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gradBias1.data(), d_db1, hiddenSize * sizeof(float), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_dZ2)); CUDA_CHECK(cudaFree(d_hidden)); CUDA_CHECK(cudaFree(d_w2));
    CUDA_CHECK(cudaFree(d_X)); CUDA_CHECK(cudaFree(d_w1));
    CUDA_CHECK(cudaFree(d_dW2)); CUDA_CHECK(cudaFree(d_db2));
    CUDA_CHECK(cudaFree(d_dW1)); CUDA_CHECK(cudaFree(d_db1));
    CUDA_CHECK(cudaFree(d_dHidden)); CUDA_CHECK(cudaFree(d_dummy));
}
