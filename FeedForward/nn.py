import numpy as np
import tensorflow as tf

# feed forward neural network
class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons): 
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()

# cross entropy loss function
class Loss_CategoricalCrossentropy:
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return np.mean(negative_log_likelihoods)
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = (dvalues - y_true) / samples

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.layer1 = Layer_Dense(input_size, hidden_size)
        self.activation1 = Activation_ReLU()
        self.layer2 = Layer_Dense(hidden_size, output_size)
        self.activation2 = Activation_Softmax()
        self.loss_function = Loss_CategoricalCrossentropy()
        self.learning_rate = learning_rate

    def train(self, X_train, y_train, epochs, batch_size):
        for epoch in range(epochs):
            batches = self._get_batches(X_train, y_train, batch_size)
            epoch_loss = 0
            epoch_accuracy = 0
            
            for X_batch, y_batch in batches:
                # forward pass
                self.layer1.forward(X_batch)
                self.activation1.forward(self.layer1.outputs)
                self.layer2.forward(self.activation1.outputs)
                self.activation2.forward(self.layer2.outputs)
                
                # loss function
                loss = self.loss_function.forward(self.activation2.outputs, y_batch)
                epoch_loss += loss
                
                # model accuracy
                predictions = np.argmax(self.activation2.outputs, axis=1)
                accuracy = np.mean(predictions == y_batch)
                epoch_accuracy += accuracy
                
                # backpropagation
                self.loss_function.backward(self.activation2.outputs, y_batch)
                self.activation2.backward(self.loss_function.dinputs)
                self.layer2.backward(self.activation2.dinputs)
                self.activation1.backward(self.layer2.dinputs)
                self.layer1.backward(self.activation1.dinputs)
                
                # update weights and biases
                self.layer1.weights -= self.learning_rate * self.layer1.dweights
                self.layer1.biases -= self.learning_rate * self.layer1.dbiases
                self.layer2.weights -= self.learning_rate * self.layer2.dweights
                self.layer2.biases -= self.learning_rate * self.layer2.dbiases
            
            
            print(f'Epoch {epoch}, loss: {epoch_loss / len(batches):.3f}, accuracy: {epoch_accuracy / len(batches):.3f}')
    
    def _get_batches(self, X, y, batch_size):
        batches = []
        for i in range(0, len(X), batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            batches.append((X_batch, y_batch))
        return batches

    def save(self, path):
        np.savez(path, 
                 layer1_weights=self.layer1.weights, layer1_biases=self.layer1.biases,
                 layer2_weights=self.layer2.weights, layer2_biases=self.layer2.biases)

    def load(self, path):
        data = np.load(path)
        self.layer1.weights = data['layer1_weights']
        self.layer1.biases = data['layer1_biases']
        self.layer2.weights = data['layer2_weights']
        self.layer2.biases = data['layer2_biases']

# data normalization
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train.reshape(-1, 784) / 255.0
X_test = X_test.reshape(-1, 784) / 255.0

nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10, learning_rate=0.05)
nn.train(X_train, y_train, epochs=51, batch_size=16)

try:
  nn.save('fashion_mnist_model.npz')
  print('model saved successfully..........')
except Exception as e:
  print(f'model not saved {e}')


'''

nn_test = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)
nn_test.load('fashion_mnist_model.npz')
print("Model loaded successfully.")

nn_test.layer1.forward(X_test)
nn_test.activation1.forward(nn_test.layer1.outputs)
nn_test.layer2.forward(nn_test.activation1.outputs)
nn_test.activation2.forward(nn_test.layer2.outputs)

predictions = np.argmax(nn_test.activation2.outputs, axis=1)
accuracy = np.mean(predictions == y_test)
print(f"Test Accuracy: {accuracy:.4f}")

'''

