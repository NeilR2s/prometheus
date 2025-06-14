"""
https://en.wikipedia.org/wiki/Linear_regression
"""

import jax.numpy as jnp

class LinearRegression:
    def __init__(self, lr = 0.001, epochs=10):

        if lr <=0: 
            raise ValueError("the learning rate must be set to a value greater than 0. ")

        if epochs <= 0:
            raise ValueError("the number of training epochs must be greater than 0. ")
        
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):

        self.X_train = jnp.asarray(X_train)
        self.y_train = jnp.asarray(y_train)

        if len(X_train) != len(y_train):
            raise ValueError(f"ensure that X and y train values are of the same length, attempted to pass arrays of incompatible length ({X_train_len},{y_train_len})")
        else:
            num_samples, num_features = jnp.shape(X_train)
        
        self.weights = jnp.zeros(num_features)
        self.bias = 0

        for _ in range(self.epochs):

            y_pred = jnp.dot(X_train, self.weights) + self.bias

            derivative_weights = (1/num_samples) * jnp.dot(X_train, (y_pred-y_train)) 
            derivative_bias = (1/num_samples) * jnp.sum(y_pred, y_train)

            self.weights = self.weights - (self.lr * self.derivative_weights)
            self.bias = self.bias - (self.lr * self.derivative_bias)

    def predict(self, y_train):

        y_train = jnp.asarray(y_train)
        y_pred = jnp.dot(y_train, self.weights) + self.bias
        
        return y_pred


