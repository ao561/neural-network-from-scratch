import numpy as np
import time

class Neural_Network:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        # Initialise weights and biases
        
        sf = 0.01 # scaling factor

        # Input to 1st HL
        self.weights1 = np.random.randn(input_size, hidden1_size) * sf
        self.bias1 = np.zeros((1, hidden1_size))

        # 1st HL to 2nd HL
        self.weights2 = np.random.randn(hidden1_size, hidden2_size) * sf
        self.bias2 = np.zeros((1, hidden2_size))

        # 2nd HL to Output
        self.weights3 = np.random.randn(hidden2_size, output_size) * sf
        self.bias3 = np.zeros((1, output_size))

    def relu(self, x):
        # ReLu activation function
        return np.maximum(0, x)
    
    def softmax(self, x):
        # softmax activation function (numerically stable)
        exp_x = np.exp(x - np.max(x, axis = 1, keepdims = True))
        return exp_x / np.sum(exp_x, axis = 1, keepdims = True)
    
    def relu_derivative(self, x):
        # ReLu derivative function
        return np.where(x > 0, 1, 0)
    
    def forward_pass(self, X):

        # Pass through 1st HL
        self.hidden1_input = np.dot(X, self.weights1) + self.bias1
        self.hidden1_output = self.relu(self.hidden1_input)

        # Pass through 2nd HL
        self.hidden2_input = np.dot(self.hidden1_output, self.weights2) + self.bias2
        self.hidden2_output = self.relu(self.hidden2_input)

        # Output
        self.final_input = np.dot(self.hidden2_output, self.weights3) + self.bias3
        self.final_output = self.softmax(self.final_input)

        return self.final_output
    
    def prediction(self, X):
        probabilities = self.forward_pass(X)
        # returns max probability from softmax
        return np.argmax(probabilities, axis = 1)

    def backpropagation(self, X, y, output, learning_rate):
        delta = output - y

        hidden2_error = delta.dot(self.weights3.T)
        hidden2_delta = hidden2_error * self.relu_derivative(self.hidden2_input)
        
        hidden1_error = hidden2_delta.dot(self.weights2.T)
        hidden1_delta = hidden1_error * self.relu_derivative(self.hidden1_input)
        
        # update weights and biases
        m = len(X)
        self.weights3 -= np.dot(self.hidden2_output.T, delta) * learning_rate / m
        self.bias3 -= np.sum(delta, axis = 0, keepdims = True) * learning_rate / m
        self.weights2 -= np.dot(self.hidden1_output.T, hidden2_delta) * learning_rate / m
        self.bias2 -= np.sum(hidden2_delta, axis = 0, keepdims = True) * learning_rate / m
        self.weights1 -= np.dot(X.T, hidden1_delta) * learning_rate / m
        self.bias1 -= np.sum(hidden1_delta, axis = 0, keepdims = True) * learning_rate / m
    
    def train(self, X, y, epochs, learning_rate, batch_size = 64):
        for epoch in range(epochs):
            start_time = time.time()

            # shuffle the training data
            permutation = np.random.permutation(len(X))
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # iterate over mini-batches
            for i in range(0, len(X), batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                output = self.forward_pass(X_batch)
                self.backpropagation(X_batch, y_batch, output, learning_rate)

            # evaluate loss, accuracy, time for pass
            full_output = self.forward_pass(X)
            loss = -np.sum(y * np.log(full_output + 1e-9)) / len(y) # prevents log error
            accuracy = self.calculate_accuracy(full_output, y)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}, Time: {epoch_time:.2f}s")
    
    def calculate_accuracy(self, output, y):
        predictions = np.argmax(output, axis = 1)
        true_labels = np.argmax(y, axis = 1)
        return np.mean(predictions == true_labels)
    
def one_hot_encode(y, num_classes):
    # creates a one hot encoded matrix of label data
    return np.eye(num_classes)[y]

if __name__ == '__main__':
    # load training and test data
    df = np.load('mnist.npz')
    x_train, y_train = df['x_train'], df['y_train']
    x_test, y_test = df['x_test'], df['y_test']

    # transform images into 784 dimension vectors
    x_train = x_train.reshape(x_train.shape[0], 784)
    x_test = x_test.reshape(x_test.shape[0], 784)

    # normalise pixel values to between 0-1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # one hot encode labels
    y_train = one_hot_encode(y_train, 10)
    y_test_original_labels = y_test # keep original labels for example predictions
    y_test = one_hot_encode(y_test, 10)

    # initialise neural network (128, 256 neurons for hidden layers)
    nn = Neural_Network(input_size = 784, hidden1_size = 256, hidden2_size = 128, output_size = 10)

    # training on dataset
    print("Starting training on dataset...")
    nn.train(x_train, y_train, epochs = 10, learning_rate = 0.1, batch_size = 64)
    print("Training finished.\n")

    # test set evaluation
    print("Evaluating on testset...")
    test_output = nn.forward_pass(x_test)
    test_accuracy = nn.calculate_accuracy(test_output, y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # example predictions
    print("\nExample predictions...")
    predictions = nn.prediction(x_test[:10])
    print(f"Predicted labels: {predictions}")
    print(f"True labels: {y_test_original_labels[:10]}")