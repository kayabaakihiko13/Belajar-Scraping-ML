import numpy as np
# https://open.spotify.com/album/4koqulx259upSRxNzOx0RT
def costFunction(x_data:np.array
                 ,y_data,theta:np.array):
    hypo = np.dot(x_data,theta)
    hypo -= y_data
    sum_error=np.sum(hypo ** 2)
    error= sum_error/(2*x_data.shape[0])
    return error

def gradientDencest(x_data:np.array,
                    y_data:np.array,
                    theta:np.array,
                    alpha:float):
    predict = np.dot(x_data,theta)
    predict -= y_data
    sum_gradient = np.dot(x_data.T,predict)
    theta = theta - (alpha/x_data.shape[0]) *sum_gradient
    return theta

def linear_regession(x_data: np.array,y_data: np.array,
                     alpha :float=0.00001,
                     iterable :int=1000,
                     intrcept :bool=True):
    X = np.array(x_data)
    y = np.array(y_data) 
    if intrcept:
        X = np.c_[np.ones((X.shape[0], 1)), X]
    _,m=X.shape
    theta=np.zeros((m,1))
    for i in range (iterable) :
        theta=gradientDencest(X,y,theta,alpha)
        error=costFunction(X,y,theta)
        print(f'{i+1}:error{error}')
    return theta

if __name__ == "__main__":
    x = np.array([[21, 2, 1, 2],
                  [12, 3, 4, 2]])
    y = np.array([[2], [2]])
    linear_regession(x, y)

    