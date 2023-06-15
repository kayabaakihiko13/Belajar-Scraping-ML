import numpy as np

def simple_linear_regession(x_data:np.array,
                     y_data:np.array,
                     learning_path=.000001,
                     itterable=1000):
    m=b=0
    for _ in range(itterable):
        predict = m* x_data + b
        predict -= y_data
        cost = sum(predict**2)/(2 * x_data.shape[0])
        # jadikan gradient distance
        md = -(2/x_data.shape[0]) * sum(x_data*(predict-y_data)) 
        bd = -(2/x_data.shape[0]) * sum((predict-y_data))
        m = m - learning_path * md
        b = b - learning_path * bd
        print(cost,m,b)


if __name__ == "__main__":
    x_data=np.array([1,2,3,4,2,1,3,3])
    y_data=np.array([1,2,3,4,2,1,3,3])
    simple_linear_regession(x_data,y_data)