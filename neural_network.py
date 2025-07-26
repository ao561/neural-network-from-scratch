import numpy as np

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
        self.hidden1_input = np.dot(X, self.W1) + self.b1
        self.hidden1_output = self.relu(self.hidden1_input)

        # Pass through 2nd HL
        self.hidden2_input = np.dot(self.hidden1_output, self.W2) + self.b2
        self.hidden2_output = self.relu(self.hidden2_input)

        # Output
        self.final_input = np.dot(self.hidden2_output, self.W3) + self.b3
        self.final_output = self.softmax(self.final_input)

        return self.final_output


    def backpropagation(self, X, y, output, learning_rate):
        delta = output - y

        hidden2_error = delta.dot(self.W3.T)
        hidden2_delta = hidden2_error * self.relu_derivative(self.hidden2_input)
        
        hidden1_error = hidden2_delta.dot(self.W2.T)
        hidden1_delta = hidden1_error * self.relu_derivative(self.hidden1_input)
        
        # update weights and biases
        m = len(X)
        self.W3 -= np.dot(self.hidden2_output.T, delta) * learning_rate / m
        self.b3 -= np.sum(delta, axis = 0, keepdims = True) * learning_rate / m
        self.W2 -= np.dot(self.hidden1_output.T, hidden2_delta) * learning_rate / m
        self.b2 -= np.sum(hidden2_delta, axis = 0, keepdims = True) * learning_rate / m
        self.W1 -= np.dot(X.T, hidden1_delta) * learning_rate / m
        self.b1 -= np.sum(hidden1_delta, axis = 0, keepdims = True) * learning_rate / m
    


    def predict(self, X):
        probabilities = self.forward_pass(X)
        return np.argmax(probabilities, axis = 1)