import numpy as np

class linear_regression:
    def __init__(self, iterations=1000, learning_rate=0.01, fit_intercept=True):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.fit_intercept = fit_intercept
        self.theta = None

    def __cost_function(self, X, y):
        m = len(y)
        h = np.dot(X, self.theta)
        error = (1 / (2 * m)) * np.sum(np.square(h - y))
        return error

    def __gradient_descent(self, X, y):
        m = len(y)
        h = np.dot(X, self.theta)
        gradient = (1 / m) * np.dot(X.T, (h - y))
        self.theta = self.theta - self.learning_rate * gradient
        return self.theta

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.concatenate((ones, X), axis=1)

        self.theta = np.zeros(X.shape[1])

        for _ in range(self.iterations):
            self.theta=self.__gradient_descent(X, y)
            cost = self.__cost_function(X, y)
            print(f"Cost: {cost}")

        return self

    def predict(self, X):
        X = np.array(X)

        if self.fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.concatenate((ones, X), axis=1)

        return np.dot(X, self.theta)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    x = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(x, np.array([1, 2])) + 3

    model = linear_regression(iterations=1000, learning_rate=0.01)
    model.fit(x, y)

    x_new = np.array([[1, 1]])
    y_pred = model.predict(x)
    
    print("Predicted:", y_pred)
