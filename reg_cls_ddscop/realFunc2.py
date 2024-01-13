import numpy as np
from smt.sampling_methods import LHS





def rosenbrock(x):
    label = []
    d = x.shape[1]
    cons = np.zeros((x.shape[0], 2))

    # 计算目标值
    f = np.sum(100.0*(x[:,1:]-x[:,:-1]**2.0)**2.0 + (1-x[:,:-1])**2.0, axis = 1)

    # 计算约束值与标签
    for i in range(x.shape[0]):
        cons[i, 0] = 9 * d - np.sum(x[i, :] ** 2)
        cons[i, 1] = np.sum(x[i, :int(d / 2)] ** 2) / int(d / 2) - (np.sum(x[i, int(d / 2):] ** 2) / (d - int(d / 2)))

        if (cons[i,0]<= 0 and cons[i,1]<= 0):
            label.append(0)
        else:
            label.append(1)
            
    return f,cons,np.array(label)



def Griewank(x):
    t = 1
    for i in range(x.shape[1]):
        t *= np.cos(x[:,i]/np.sqrt(i+1))
    f = 1 + np.sum((x**2)/4000, axis = 1) - t

    return f







if __name__ == "__main__":
    xlimits, num = np.array([[-600,6]] * 10), 1000
    sampling = LHS(xlimits=xlimits, criterion="cm")
    X = sampling(num)
    print((rosenbrock(X)[2] == 0).sum())










#Rosenbrock Problem
#def Rosenbrock01(x):
#Griewank Problem