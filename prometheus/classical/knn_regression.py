import jax
import jax.numpy as jnp
from prometheus.utils.distances import manhattan_distance, euclidean_distance, minkowski_distance

# TODO : Implement a label encoder 

class KNN_Regression:
    """
    A K-Nearest Neighbors (KNN) regressor implemented using JAX.

    1. Calculate distance from all datapoints in the dataset
    2. Get closest K points
    3. Get average of K points

    Args:
        - k (int): The number of nearest neighbors to consider for classification. Default is 3.
        - X_train (jnp.ndarray): The training data features, stored after calling fit().
        - y_train (jnp.ndarray): The training data labels, stored after calling fit().
        - distance_metric (callable): The function used to calculate distances between points. Options are euclidean, minkowski, manhattan, but default is euclidean.
    """
    def __init__(self, k: int = 3, distance_metric: str = "euclidean", minkowski_q: int = 2):
        """
        Initializes the KNN_Classifier.

        Args:
            - k (int): The number of nearest neighbors. Defaults to 3.
            - distance_metric (str): The distance metric to use. Options: 'euclidean', 'manhattan', 'minkowski'. Defaults to 'euclidean'.
            - minkowski_q (int): The 'q' parameter for the Minkowski distance, only used if distance_metric is 'minkowski'. Defaults to 2 (Euclidean).

        Raises:
            - ValueError: If an unsupported distance_metric is provided or if minkowski_q is invalid for 'minkowski' distance.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        
        if k <= 0:
            raise ValueError("k must be a positive integer")

        if distance_metric == "euclidean":
            self.distance_metric = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_metric = manhattan_distance
        elif distance_metric ==  "minkowski":
            self.distance_metric = lambda minkowski_distance: minkowski_distance(q=minkowski_q)

        else:
            raise ValueError(f"unsupported distance metric {distance_metric}, please choose between euclidean, manhattan, or minkowski")


    def fit(self, X, y) -> list:
        """
        Stores the training feature data (X) and corresponding labels (y). It casts input X and y into JAX arrays.

        Args:
            X (array-like): Training data features - 2D array-like structure where rows are samples and columns are features.
                            Must be convertible to a JAX array of numerical data.
            y (array-like): Training data labels - 1D array-like structure corresponding to the rows in X.
        """

        self.X_train = jnp.asarray(X)
        self.y_train = jnp.asarray(y)

        X_train_len = self.X_train.shape[0] 
        y_train_len = self.y_train.shape[0]

        if self.X_train.ndim == 1:
            self.X_train.reshape(-1,1)
        if X_train_len != y_train_len:
            raise ValueError(f"ensure that the X and y train values are of the same length, attempted to pass ({X_train_len},{y_train_len})")
        if self.k > X_train_len:
            raise ValueError("k must not be greater than the number of training samples")

    def predict(self, X_test: jnp.ndarray) -> jnp.array:
        """
        Args:
            X_test (array-like): Test data features - 2D array-like structure where rows are samples and columns are features.

        Returns:
            predictions (list): A list of predicted labels for each sample in X_test.

        Raises:
            ValueError: If the model has not been fitted yet 
        
        """
        if self.X_train == None or self.y_train == None:
            raise ValueError("the KNN model must be fit with data before predicting")

        X_test = jnp.asarray(X_test)
        X_train_features = self.X_train.shape[1]
        X_test_features = X_test.shape[1]

        if X_test.ndim == 1:
            X_test.reshape(1,-1)
        if X_train_features != X_test_features:
            raise ValueError(f"the KNN model was trained on {X_train_features} but the testing data has {X_test_features}") 

        predictions = jnp.asarray([self._predict(X_test_sample) for X_test_sample in X_test])
        return predictions

    def _predict(self, x_test_sample: jnp.ndarray) -> jnp.array:
        """
        Predicts the class label for a single test sample.

        Args:
            x_test_sample (jnp.ndarray): A single sample from the test data (a 1D JAX array).

        Returns:
            The predicted class label for the input sample.
        """

        distances = jnp.asarray([self.distance_metric(x_test_sample, X_train_sample) for X_train_sample in self.X_train])
        # print(f"distances: \n{distances}")
        top_k_indices = jnp.argsort(distances)[:self.k]
        # print(f"top_k_indices: \n{top_k_indices}")
        top_k_labels = jnp.asarray([self.y_train[i] for i in top_k_indices])
        # print(f"top k labels: \n{top_k_labels}")

        return jnp.mean(top_k_labels)