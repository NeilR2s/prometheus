from functools import partial
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import jit
from prometheus.utils.distances import manhattan_distance, euclidean_distance, minkowski_distance


def _fit(X_train: ArrayLike, y_train:ArrayLike, k: int):

    X_train = jnp.asarray(X_train)
    y_train = jnp.asarray(y_train)

    X_train_len = X_train.shape[0] 
    y_train_len = y_train.shape[0]

    # we expect a 2d array, reshaping 1d array to cover edge case
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1,1)
    if X_train_len != y_train_len:
        raise ValueError(f"ensure that the X and y train values are of the same length, attempted to pass arrays of incompatible length ({X_train_len},{y_train_len})")
    if k > X_train_len:
        raise ValueError("k must not be greater than the number of training samples")

    return X_train, y_train

# expensive operation is jit compiled, partially filled with functools to enable compilation
@partial(jax.jit, static_argnames=("k", "distance_metric"))
def _predict(X_test: jnp.ndarray, X_train: jnp.ndarray, y_train: jnp.ndarray, k: int, distance_metric: callable) -> jnp.ndarray:
    """
    Predicts the class label for a single test s    ample.

    Args:
        x_test_sample (jnp.ndarray): A single sample from the test data (a 1D JAX array).

    Returns:
        The predicted class label for the input sample.
    """

    distances = jax.vmap(distance_metric, in_axes=(None, 0))(X_test, X_train)
    top_k_indices = jnp.argsort(distances)[:k]
    top_k_labels = y_train[top_k_indices]

    # MAJORITY VOTE, NOT GREEDY SELECTION
    unique_labels, counts = jnp.unique(top_k_labels, return_counts=True, size=top_k_labels.shape[0])
    max_label_index = jnp.argmax(counts)
    most_frequent_label = unique_labels[max_label_index]
    
    return most_frequent_label

# high-level OOP-style wrapper to easily interface with our functional-style JAX core functions
class KNN_Classifier:
    """
    A K-Nearest Neighbors (KNN) classifier implemented using JAX.

    1. Calculate distance from all datapoints in the dataset
    2. Get closest K points
    3. Get label with majority vote
    """
    def __init__(self, k: int = 3, distance_metric: str = "euclidean", minkowski_q: int = 2):
        """
        Args:
            - k (int): The number of nearest neighbors. Defaults to 3.
            - distance_metric (str): The distance metric to use. Options: 'euclidean', 'manhattan', 'minkowski'. Defaults to 'euclidean'.
            - minkowski_q (int): The 'q' parameter for the Minkowski distance, only used if distance_metric is 'minkowski'. Defaults to 2 (Euclidean).

        Raises:
            - ValueError: If an unsupported distance_metric is provided or if minkowski_q is invalid for 'minkowski' distance.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer")

        self.k = k
        self.minkowski_q = minkowski_q
        if distance_metric == "euclidean":
            self.distance_metric = euclidean_distance
        elif distance_metric == "manhattan":
            self.distance_metric = manhattan_distance
        elif distance_metric ==  "minkowski":
            self.distance_metric = lambda x_array_like, y_array_like: minkowski_distance(x=x_array_like, y=y_array_like, q=int(self.minkowski_q))
        else:
            raise ValueError(f"unsupported distance metric {distance_metric}, please choose between euclidean, manhattan, or minkowski")

        self.x_train_data_state = None
        self.y_train_data_state = None

    @property
    def info(self):
        pass

    def fit(self, X_train: ArrayLike, y_train: ArrayLike, k:int):
        """
        Stores the training feature data (X) and corresponding labels (y). It casts input X and y into JAX arrays.

        Args:
            X_train (array-like): Training data features - 2D array-like structure where rows are samples and columns are features.
                            Must be convertible to a JAX array of numerical data.
            y_train (array-like): Training data labels - 1D array-like structure corresponding to the rows in X.
        """

        self.x_train_data_state, self.y_train_data_state = _fit(X_train, y_train, self.k)

    def predict(self, X_test: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            X_test (array-like): Test data features - 2D array-like structure where rows are samples and columns are features.

        Returns:
            predictions (list): A list of predicted labels for each sample in X_test.

        Raises:
            RuntimeError: If the model has not been fitted yet 
        """

        if self.x_train_data_state is None:
            raise RuntimeError("The KNN model must be fit with data before predicting. Call .fit() first.")

        X_test = jnp.asarray(X_test)
        X_test_features = X_test.shape[1]

        if X_test.ndim == 1:
            X_test = X_test.reshape(1,-1)

        # predictions = _predict(X_test, self.x_train_data_state, self.y_train_data_state, self.k, self.distance_metric)

        predictions = jax.vmap(_predict, in_axes=(0, None, None, None, None))(X_test, self.x_train_data_state, self.y_train_data_state, self.k, self.distance_metric)

        return predictions
