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


    # def backpropagation(self, X, y, output, learning_rate):
    


    def predict(self, X):
        probabilities = self.forward_pass(X)
        return np.argmax(probabilities, axis = 1)