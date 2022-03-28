import numpy as np

def T_2_test(X , mu_0):
    '''
    返回X对应mu_0的T方统计量的值
    '''
    n , m = X.shape
    S = np.cov(X.T) #*求出样本协方差矩阵
    bar_x = np.average(X , axis = 0)
    T_2 = n * np.matmul( np.matmul(bar_x - mu_0 , np.linalg.inv(S)) , (bar_x - mu_0).T)
    return T_2 #* 返回对应的T^2统计量