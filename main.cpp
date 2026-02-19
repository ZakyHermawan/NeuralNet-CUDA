#include <random>
#include <memory>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cstddef>
#include <numeric>
#include <execution>

#include "kernel.cuh"

class NeuralNetwork
{
	std::vector<float> m_weights1;
	std::vector<float> m_bias1;
	std::vector<float> m_weights2;
	std::vector<float> m_bias2;
	std::vector<float> m_gradWeights1;
	std::vector<float> m_gradWeights2;
	std::vector<float> m_gradBias1;
	std::vector<float> m_gradBias2;
	std::vector<float> m_hiddenLayer;

	size_t m_inputSize;
	size_t m_hiddenSize;
	size_t m_outputSize;

	std::mt19937 m_gen;
public:
	NeuralNetwork(size_t inputSize, size_t hiddenSize, size_t outputSize, size_t maxBatchSize)
		: m_inputSize(inputSize), m_hiddenSize(hiddenSize), m_outputSize(outputSize),
		  m_gen(std::random_device{}())
	{
		m_weights1.resize(m_inputSize * m_hiddenSize);
		m_bias1.resize(m_hiddenSize);

		m_weights2.resize(m_hiddenSize * m_outputSize);
		m_bias2.resize(m_outputSize);

		m_gradWeights1.resize(m_inputSize * m_hiddenSize);
		m_gradBias1.resize(m_hiddenSize);

		m_gradWeights2.resize(m_hiddenSize * m_outputSize);
		m_gradBias2.resize(m_outputSize);

		m_hiddenLayer.resize(maxBatchSize * m_hiddenSize);
		initializeRandomData();
	}

	void initializeRandomData()
	{
		float scale1 = std::sqrt(2.0f / m_inputSize);
		float scale2 = std::sqrt(2.0f / m_hiddenSize);

		std::normal_distribution<float> dist1(0.0f, scale1);
		std::normal_distribution<float> dist2(0.0f, scale2);

		std::generate(m_weights1.begin(), m_weights1.end(), [&]() {
			return dist1(m_gen);
		});

		std::generate(m_weights2.begin(), m_weights2.end(), [&]() {
			return dist2(m_gen);
		});

		std::fill(m_bias1.begin(), m_bias1.end(), 0.0f);
		std::fill(m_bias2.begin(), m_bias2.end(), 0.0f);
	}

	std::vector<float> linearForward(std::vector<float>& inputData, size_t batchSize)
	{
		return forwardPipelineWrapper(
			inputData,
			m_weights1, m_bias1,
			m_weights2, m_bias2,
			m_hiddenLayer,
			batchSize, m_inputSize, m_hiddenSize, m_outputSize
		);
	}

	void backward(const std::vector<float>& batch_X, const std::vector<float>& dZ2, size_t batchSize, float learningRate)
	{
		backpropPipelineWrapper(
			dZ2, m_hiddenLayer, m_weights2,
			batch_X, m_weights1,
			m_gradWeights2, m_gradBias2,
			m_gradWeights1, m_gradBias1,
			batchSize, m_inputSize, m_hiddenSize, m_outputSize
		);

		// Apply the gradients to weights (W = W - lr * grad)
		updateWeights(learningRate);
	}

	void updateWeights(float learningRate)
	{
		for (size_t i = 0; i < m_weights1.size(); ++i) m_weights1[i] -= learningRate * m_gradWeights1[i];
		for (size_t i = 0; i < m_bias1.size(); ++i)    m_bias1[i] -= learningRate * m_gradBias1[i];
		for (size_t i = 0; i < m_weights2.size(); ++i) m_weights2[i] -= learningRate * m_gradWeights2[i];
		for (size_t i = 0; i < m_bias2.size(); ++i)    m_bias2[i] -= learningRate * m_gradBias2[i];
	}

	void saveWeights(const std::string& filename) const
	{
		std::ofstream file(filename, std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("Could not open file for saving weights: " + filename);
		}

		// Save the architecture dimensions (Safety Check)
		file.write(reinterpret_cast<const char*>(&m_inputSize), sizeof(size_t));
		file.write(reinterpret_cast<const char*>(&m_hiddenSize), sizeof(size_t));
		file.write(reinterpret_cast<const char*>(&m_outputSize), sizeof(size_t));

		// Save the raw float data for weights and biases
		file.write(reinterpret_cast<const char*>(m_weights1.data()), m_weights1.size() * sizeof(float));
		file.write(reinterpret_cast<const char*>(m_bias1.data()), m_bias1.size() * sizeof(float));
		file.write(reinterpret_cast<const char*>(m_weights2.data()), m_weights2.size() * sizeof(float));
		file.write(reinterpret_cast<const char*>(m_bias2.data()), m_bias2.size() * sizeof(float));

		std::cout << "[Success] Model saved to " << filename << std::endl << std::endl;
	}

	void loadWeights(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open())
		{
			throw std::runtime_error("Could not open file for loading weights: " + filename);
		}

		// Read and verify architecture dimensions
		size_t loadedInput, loadedHidden, loadedOutput;
		file.read(reinterpret_cast<char*>(&loadedInput), sizeof(size_t));
		file.read(reinterpret_cast<char*>(&loadedHidden), sizeof(size_t));
		file.read(reinterpret_cast<char*>(&loadedOutput), sizeof(size_t));

		if (loadedInput != m_inputSize || loadedHidden != m_hiddenSize || loadedOutput != m_outputSize)
		{
			throw std::runtime_error("Architecture mismatch! The saved model does not match the current network.");
		}

		// Load the raw float data directly into our vectors
		file.read(reinterpret_cast<char*>(m_weights1.data()), m_weights1.size() * sizeof(float));
		file.read(reinterpret_cast<char*>(m_bias1.data()), m_bias1.size() * sizeof(float));
		file.read(reinterpret_cast<char*>(m_weights2.data()), m_weights2.size() * sizeof(float));
		file.read(reinterpret_cast<char*>(m_bias2.data()), m_bias2.size() * sizeof(float));

		std::cout << "[Success] Model loaded from " << filename << std::endl;
	}
};

/*
 * ============================================================================
 * CATEGORICAL CROSS-ENTROPY LOSS (Batched)
 * ============================================================================
 * 1    N
 * L =  - ---  SUM [ log(y_hat_{j, correct}) ]
 * N   j=1
 *
 * Where:
 * - L      : The average loss for the entire batch.
 * - N      : The batch size.
 * - y_hat  : The predicted probability for the CORRECT class.
 * * Note: We add a tiny epsilon (1e-7) to the probability before taking the
 * log to prevent std::log(0), which would result in negative infinity (NaN).
 * ============================================================================
 */
static float computeCrossEntropyLossBatched(const std::vector<float>& probabilities, const std::vector<int>& targets, size_t batchSize, size_t numClasses)
{
	float total_loss = 0.0f;

	for (size_t i = 0; i < batchSize; ++i)
	{
		int target_class = targets[i]; // E.g., The true digit is '7'

		// Find the probability the network guessed for '7'
		float correct_class_prob = probabilities[i * numClasses + target_class];

		// Add a tiny number (1e-7f) to prevent std::log(0) which crashes the program
		total_loss += -std::log(correct_class_prob + 1e-7f);
	}

	return total_loss / static_cast<float>(batchSize);
}

/*
 * ============================================================================
 * SOFTMAX EQUATION (with numerical stability shift)
 * ============================================================================
 * * exp(z_i - max(z))
 * Softmax(z_i) = -----------------------
 * sum_{j=1}^{K} exp(z_j - max(z))
 * * Where:
 * - z_i    : The raw prediction score (logit) for a specific class 'i'.
 * - max(z) : The highest logit in the array (prevents overflow to infinity).
 * - K      : The total number of classes (e.g., 10 for digits 0-9).
 * * Note: This function processes a flattened 1D vector containing a batch of
 * multiple images. It computes the Softmax independently for each image (row).
 * ============================================================================
 */
static std::vector<float> computeSoftMaxBatched(const std::vector<float>& logits, size_t numClasses)
{
	if (logits.empty() || numClasses == 0) return {};

	std::vector<float> result(logits.size());
	size_t batchSize = logits.size() / numClasses;

	// Loop through each image one by one
	for (size_t i = 0; i < batchSize; ++i)
	{
		size_t offset = i * numClasses;

		float max_val = logits[offset];
		for (size_t j = 1; j < numClasses; ++j)
		{
			if (logits[offset + j] > max_val) max_val = logits[offset + j];
		}

		float sum_exp = 0.0f;
		for (size_t j = 0; j < numClasses; ++j)
		{
			result[offset + j] = std::exp(logits[offset + j] - max_val);
			sum_exp += result[offset + j];
		}

		for (size_t j = 0; j < numClasses; ++j)
		{
			result[offset + j] /= sum_exp;
		}
	}

	return result;
}

template<typename T>
std::vector<T> readBinaryFile(std::string filePath, size_t N)
{
	std::ifstream file(filePath, std::ios::binary);
	if (!file)
	{
		std::cout << "?\n";
		throw std::runtime_error("Couldn't open file " + filePath);
	}

	std::vector<T> buffer(N);
	file.read(reinterpret_cast<char*>(buffer.data()), N * sizeof(T));
	return buffer;
}

static int inferenceMode(float mean, float std_dev)
{
	size_t INPUT_SIZE = 784; // 28 x 28 pixels
	size_t HIDDEN_SIZE = 1024;
	size_t OUTPUT_SIZE = 10; // 10 digits
	size_t TEST_SIZE = 1000;
	size_t BATCH_SIZE = 32;

	std::vector<float> X_test = readBinaryFile<float>("data/X_test.bin", TEST_SIZE * INPUT_SIZE);
	std::vector<int> y_test = readBinaryFile<int>("data/y_test.bin", TEST_SIZE);
	NeuralNetwork nn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

	try
	{
		nn.loadWeights("E:/SourceCodes/NeuralNet-Implementations/NeuralNet-Implementations/best_mnist_model.bin");
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return -1;
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> distr(0, 999);
	size_t image_index = distr(gen);

	auto start_it = X_test.begin() + (image_index * INPUT_SIZE);
	std::vector<float> single_image(start_it, start_it + INPUT_SIZE);
	int true_label = y_test[image_index];

	NormalizeCUDA(single_image, (float)mean, (float)std_dev);

	std::vector<float> logits = nn.linearForward(single_image, 1);
	std::vector<float> probabilities = computeSoftMaxBatched(logits, OUTPUT_SIZE);

	float best_prob = -1.0f;
	int predicted_digit = -1;
	for (size_t c = 0; c < OUTPUT_SIZE; ++c)
	{
		if (probabilities[c] > best_prob)
		{
			best_prob = probabilities[c];
			predicted_digit = c;
		}
	}

	std::cout << "\n--- Inference Result ---" << std::endl;
	std::cout << "Image Index   : " << image_index << std::endl;
	std::cout << "True Reality  : " << true_label << std::endl;
	std::cout << "AI Prediction : " << predicted_digit << " (" << (best_prob * 100.0f) << "%)" << std::endl;

	if (predicted_digit == true_label) std::cout << "Result        : CORRECT!" << std::endl;
	else std::cout << "Result        : WRONG!" << std::endl;

	return 0;
}

int main()
{
	//inferenceMode(0.1307, 0.3081);
	//return 0;

	size_t INPUT_SIZE = 784; // 28 x 28 pixels
	size_t HIDDEN_SIZE = 1024;
	size_t OUTPUT_SIZE = 10; // 10 digits
	size_t TRAIN_SIZE = 60000;
	size_t TEST_SIZE = 1000;
	size_t BATCH_SIZE = 32;
	size_t EPOCHS = 10;
	float LEARNING_RATE = 0.01f;

	std::vector<float> X_train = readBinaryFile<float>("data/X_train.bin", TRAIN_SIZE * INPUT_SIZE);
	std::vector<float> X_test = readBinaryFile<float>("data/X_test.bin", TEST_SIZE * INPUT_SIZE);
	std::vector<int> y_train = readBinaryFile<int>("data/y_train.bin", TRAIN_SIZE);
	std::vector<int> y_test = readBinaryFile<int>("data/y_test.bin", TEST_SIZE);

	std::vector<float> copyXTrain(X_train);

	double sum = std::reduce(std::execution::par, X_train.begin(), X_train.end(), 0.0);
	double mean = sum / X_train.size();
	double sq_sum = std::accumulate(X_train.begin(), X_train.end(), 0.0,
		[mean](double acc, float x)
		{
			return acc + std::pow(x - mean, 2);
		});
	double std_dev = std::sqrt(sq_sum / X_train.size());
	std::cout << "Mean: " << mean << " Std: " << std_dev << std::endl;

	NormalizeCUDA(X_train, (float)mean, (float)std_dev);
	NormalizeCUDA(X_test, (float)mean, (float)std_dev);

	NeuralNetwork nn(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, BATCH_SIZE);

	float best_accuracy = 0.0f;
	for (size_t epoch = 0; epoch < EPOCHS; ++epoch)
	{
		float epoch_loss = 0.0f;
		for (size_t i = 0; i < TRAIN_SIZE; i += BATCH_SIZE)
		{
			// Handle the last batch if TRAIN_SIZE isn't perfectly divisible by BATCH_SIZE
			size_t current_batch_size = std::min(BATCH_SIZE, TRAIN_SIZE - i);
			auto start_X = X_train.begin() + (i * INPUT_SIZE);
			auto end_X = X_train.begin() + ((i + current_batch_size) * INPUT_SIZE);
			std::vector<float> batch_X(start_X, end_X);

			auto start_y = y_train.begin() + i;
			auto end_y = y_train.begin() + (i + current_batch_size);
			std::vector<int> batch_y(start_y, end_y);

			std::vector<float> y_logits = nn.linearForward(batch_X, current_batch_size);
			std::vector<float> probabilities = computeSoftMaxBatched(y_logits, OUTPUT_SIZE);
			float loss = computeCrossEntropyLossBatched(probabilities, batch_y, current_batch_size, OUTPUT_SIZE);
			epoch_loss += loss;

			// PREPARE GRADIENT FOR BACKPROPAGATION (dZ2)
			// Formula: (Probabilities - True Labels) / BatchSize
			std::vector<float> dZ2 = probabilities; // Start with a copy of the probabilities
			for (size_t b = 0; b < current_batch_size; ++b)
			{
				int correct_class = batch_y[b];

				// Raw error for the correct class: Prob - 1.0
				dZ2[b * OUTPUT_SIZE + correct_class] -= 1.0f;

				// Average the entire gradient row for the batch
				// We do this once per class to prevent the "Double-Division" bug
				float batch_scale = 1.0f / static_cast<float>(current_batch_size);
				for (size_t c = 0; c < OUTPUT_SIZE; ++c)
				{
					dZ2[b * OUTPUT_SIZE + c] *= batch_scale;
				}
			}

			nn.backward(batch_X, dZ2, current_batch_size, LEARNING_RATE);

			std::cout << "\rEpoch [" << (epoch + 1) << "/" << EPOCHS << "] "
				<< "- Training data: " << (i + current_batch_size) << "/" << TRAIN_SIZE;
		}

		float val_loss = 0.0f;
		int correct_predictions = 0;

		for (size_t i = 0; i < TEST_SIZE; i += BATCH_SIZE)
		{
			size_t current_batch_size = std::min(BATCH_SIZE, TEST_SIZE - i);
			auto start_X = X_test.begin() + (i * INPUT_SIZE);
			auto end_X = X_test.begin() + ((i + current_batch_size) * INPUT_SIZE);
			std::vector<float> batch_X(start_X, end_X);

			auto start_y = y_test.begin() + i;
			auto end_y = y_test.begin() + (i + current_batch_size);
			std::vector<int> batch_y(start_y, end_y);

			// Forward Pass Only (No backward pass during testing)
			std::vector<float> y_logits = nn.linearForward(batch_X, current_batch_size);
			std::vector<float> probabilities = computeSoftMaxBatched(y_logits, OUTPUT_SIZE);

			// Accumulate Validation Loss
			val_loss += computeCrossEntropyLossBatched(probabilities, batch_y, current_batch_size, OUTPUT_SIZE);

			// Calculate Accuracy (Argmax)
			for (size_t b = 0; b < current_batch_size; ++b)
			{
				float maxProb = -1.0f;
				int prediction = -1;

				// Find the digit with the highest probability
				for (size_t c = 0; c < OUTPUT_SIZE; ++c)
				{
					if (probabilities[b * OUTPUT_SIZE + c] > maxProb)
					{
						maxProb = probabilities[b * OUTPUT_SIZE + c];
						prediction = c;
					}
				}

				// Check if the network's prediction matches reality
				if (prediction == batch_y[b])
				{
					correct_predictions++;
				}
			}
		}

		// Calculate averages and percentages
		float avg_train_loss = epoch_loss / std::ceil(static_cast<float>(TRAIN_SIZE) / BATCH_SIZE);
		float avg_val_loss = val_loss / std::ceil(static_cast<float>(TEST_SIZE) / BATCH_SIZE);
		float val_accuracy = (static_cast<float>(correct_predictions) / TEST_SIZE) * 100.0f;

		// Print the final epoch summary on a new line!
		std::cout << "\nTrain Loss: " << avg_train_loss
			<< " | Validation Loss: " << avg_val_loss
			<< " | Validation Acc: " << val_accuracy << "%" << std::endl;

		if (val_accuracy > best_accuracy)
		{
			best_accuracy = val_accuracy;
			nn.saveWeights("best_mnist_model.bin");
		}
	}

	return 0;
}
