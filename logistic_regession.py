import numpy as np
import matplotlib.pyplot as plt

class logistik_regession:
    
    def __init__(self,iterable:int,
                 alpha:float,intercept:bool=True) -> None:
        self.iterable = iterable
        self.learning_path = alpha
        self.intercept = intercept
        self.theta = None

    
    def __sigmoid(self,z:np.ndarray):
        return 1 / (1 + np.exp(-z))


    def __costFunction(self,h:np.array,
                       y:np.ndarray):
        n = h.shape[0]
        return (-y * np.log(h) - (1 - y) *np.log(1 - h)).mean()

    
    def fit(self,x:np.array,y:np.array)-> object:
        X = np.array(x)
        y = np.array(y)
        if self.intercept:
            size = np.ones((X.shape[0], 1))
            X=np.concatenate((size,X), axis=1)
        self.theta = np.zeros(X.shape[1])
        for i in range(self.iterable):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.learning_path * gradient
            z = np.dot(X,self.theta)
            h = self.__sigmoid(z)
            loss = self.__costFunction(h, y)
            print(f"i:{i}loss:{loss}")
        return self
    def predict_proba(self, x: np.array):
        X = np.array(x)
        if self.intercept:
            size = np.ones((X.shape[0], 1))
            X=np.concatenate((size,X), axis=1)
        return self.__sigmoid(np.dot(X, self.theta))

    def prediction(self,x):
        return self.predict_proba(x).round()

if __name__ =="__main__":
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    model = logistik_regession(iterable=1000,alpha=0.1)
    model.fit(X,y)
    preds = model.prediction(X)
    # (preds == y).mean()
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='b', label='0')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='r', label='1')
    plt.legend()
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = model.predict_proba(grid).reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
    plt.show()

