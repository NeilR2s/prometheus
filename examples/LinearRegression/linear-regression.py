import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# not quite scikit-learn like, but this will do for now

class LinearRegression:

    def __init__(self):
        self.x = None
        self.y = None
        self.x_mean = None
        self.y_mean = None   
        self.x_mean_dev = None
        self.y_mean_dev = None
        self.x_sum_sqr = None
        self.y_sum_sqr = None
        self.product_sum = None
        self.pearson_r = None
        self.Sy = None
        self.Sx = None
        self.b = None
        self.A = None
        self.y_pred = None

    def fit(self, x: list, y: list):
        try:
            self.x = np.array(x, dtype = np.float32)
            self.y = np.array(y, dtype = np.float32)
            if len(self.x) != len(self.y):
                raise ValueError('Input array and and y are of unequal shape')
        except Exception as e:
            raise e

    def statistics(self):

        '''
        calculates the statistical simple regression of elements in an array of equal
        shape following the formula Y' = bX + A where:

        b: slope
        A: y intercept

        Returns:
            x_mean (np.float32): 
            y_mean (np.float32):
            b (np.float32): slope of the regression line
            pearson_r (np.float32): measure of correlation between x and y
        
        '''
        if self.x is None or self.y is None:
            raise ValueError('Fit x and y values to the model.')
        self.x_mean = np.mean(self.x)
        self.y_mean = np.mean(self.y)     
        self.x_mean_dev = self.x - self.x_mean
        self.y_mean_dev = self.y - self.y_mean
        self.x_sum_sqr = np.sum((self.x - self.x_mean)**2)
        self.y_sum_sqr = np.sum((self.y-self.y_mean)**2)
        self.product_sum = np.sum([(x*y) for x,y in zip(self.x_mean_dev, self.y_mean_dev)])
        self.pearson_r = self.product_sum / (np.sqrt(self.x_sum_sqr * self.y_sum_sqr))
        self.Sy = np.sqrt(self.y_sum_sqr/(len(self.y) - 1))
        self.Sx = np.sqrt(self.x_sum_sqr/(len(self.x) -1))
        self.b = self.pearson_r*(self.Sy/self.Sx)
        self.A = self.y_mean - self.b * self.x_mean 
        return {
            "x_mean": self.x_mean,
            "y_mean": self.y_mean,
            "slope": self.b,
            "y_intercept": self.A,
            "Pearson's r": self.pearson_r,
            "y_mean_deviation": np.std(self.y)
        }

    def predict(self):
        if self.b is None:
            raise ValueError('Compute model statistics before predicting values.')
        self.y_pred = [(self.b * x + self.A) for x in self.x]
        return self.y_pred
    
    def evaluate(self):
        if self.y_pred is None:
            raise ValueError('Compute model prediction before evaluating.')
        
        residual = self.y - self.y_pred
        sum_sqr_total = np.sum((np.square(residual)**2))
        sum_sqr_residual = np.sum((np.square(self.y_mean)**2))
        r2 = (1-(sum_sqr_residual/sum_sqr_total))
        mse = sum_sqr_residual/len(self.y)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residual))
        return {
            "R2 (Pearson's r)": r2,
            "Mean Squared Error": mse,
            "Root Mean Squared Error": rmse,
            "Mean Absolute Error": mae
        }
    def plot(self):
        plt.scatter(self.x, self.y)
        plt.plot(self.x, self.y_pred, 'red')
        plt.xlabel("Median House Value")
        plt.ylabel("Median Income")
        plt. title("Median House Value vs Median Income")
        plt.show()


df = pd.read_csv('housing.csv')
model = LinearRegression()
df = df[['median_income', 'median_house_value']]
df.drop(df[df['median_house_value']>500000].index, inplace=True)
df.plot.scatter('median_income', 'median_house_value')
X = df.iloc[:, 0].values. reshape(-1,1)
y = df. iloc[:, 1]. values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model.fit(x = X_train, y = y_train)
print(model.statistics())
model.predict()
print(model.evaluate())
model.plot()
