import numpy as np
from numpy._typing import ArrayLike
import numpy.random as nr



class SVM:

    def __init__(self, learning_rate:float= 0.01, lambda_param: float = 0.01, epochs:int=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param # i think this is for weight decay
        self.epochs = epochs
        self.weights = None
        self.biases = None

    def fit(self, X_train: ArrayLike, y_train: ArrayLike) -> None:

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        num_samples, num_features = X_train.shape

        binarized_y = np.where(y_train<=0, -1, 1)

        # key_seed = jr.PRNGKey(0)
        # self.weights = jr.uniform(key_seed, shape=num_features)

        self.weights = np.zeros(num_features)
        self.bias = 0

        # Linear Model Kernel function (y(w*xi-b)>=1)
        for _ in range(self.epochs):
            for i, x_i in enumerate(X_train): 
                decision = y_train[i] * (np.dot(x_i, self.weights) - self.bias) >=1
                if decision:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else: 
                    self.weights -= self.learning_rate * (2 * self.lambda_param - (np.dot(x_i, y_train[i])))
            
    def train(self):
        pass

    def predict(self, X_test: ArrayLike):
        approx = np.dot(X_test, self.weights) - self.bias
        prediction = np.sign(approx)

        return prediction

if __name__ == '__main__':
    X_train = [[1,2],[2,3],[3,3],[2,1]]
    y_train = [1,1,-1,-1]
    model = SVM(epochs=2)
    model.fit(X_train, y_train)
    X_test = [2.5,2]   
    prediction = model.predict(X_test)
    print(prediction)

