# Neural Network from Scratch
## 🧠 Overview
A deep neural network developed from the ground up using only NumPy for the task of handwritten digit recognition on the MNIST dataset. This project was a hands-on exploration of the core principles of deep learning, including network architecture, optimisation algorithms, and loss functions, without relying on high-level frameworks like PyTorch or TensorFlow.

## 🛠️ Key Features
* **From-Scratch Implementation**: The entire neural network, including all layers and operations, is built using only the fundamental numerical library, NumPy.

* **Handwritten Digit Recognition**: The model is trained and tested on the industry-standard MNIST dataset to classify handwritten digits (0-9).

* **High Accuracy**: The model achieves a test accuracy of over 98% on the MNIST dataset, demonstrating effective learning and generalisation.

* **Mini-Batch Gradient Descent**: The training process uses mini-batch gradient descent to efficiently update model parameters.

* **Categorical Cross-Entropy Loss**: A categorical cross-entropy loss function is implemented and used to measure the model's performance and guide the optimisation process.

* **Activation Functions**: The network uses common activation functions such as ReLU and Softmax.

## ⚙️ How to Run
### 1. Clone the repository
``` bash
git clone https://github.com/ao561/neural-network-from-scratch.git
cd neural-network-from-scratch
```
### 2. Install Dependencies:
This project only requires numpy
``` bash
pip install numpy
```
### 3. Run the Training Script
The main script will load the MNIST dataset, train the model, and evaluate its performance.
``` bash
python train.py
```
