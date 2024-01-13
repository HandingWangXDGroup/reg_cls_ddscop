import numpy as np
'''
本程序用于计算真实目标函数和真实约束值及其标签
'''

# 线性约束Ellipsoid问题
def Ellipsoid01(x):
    label = []
    d = x.shape[1]
    cons = np.zeros((x.shape[0], 2))
    #计算目标值
    f = np.zeros(x.shape[0])
    for i in range(1,d+1):
        f += i*(x[:,i-1]**2)

    #计算约束值与标签
    for i in range(x.shape[0]):
        cons[i,0] = (5 - np.sum(x[i, :]))/d
        cons[i,1] = np.sum(x[i, :int(d / 2)]) / int(d / 2) - np.sum(x[i, int(d / 2):]) / (d - int(d / 2))
        if (cons[i,0]  <= 0 and cons[i,1]  <= 0):
            label.append(0)
        else:
            label.append(1)
    return f,cons,np.array(label)


# 非线性约束Ellipsoid问题
def Ellipsoid02(x):
    label = []
    d = x.shape[1]
    f = np.zeros(x.shape[0])
    cons = np.zeros((x.shape[0], 2))

    # 计算目标值
    for i in range(1,d+1):
        f += i*(x[:,i-1]**2)

    # 计算约束值与标签
    for i in range(x.shape[0]):
        cons[i,0] = 9*d - np.sum(x[i, :]**2)
        cons[i,1] = np.sum(x[i, :int(d / 2)]**2) / int(d / 2) - (np.sum(x[i, int(d / 2):]**2) / (d - int(d / 2)))

        if (cons[i,0]<= 0 and cons[i,1]<= 0):
            label.append(0)
        else:
            label.append(1)
    return f,cons,np.array(label)


#线性约束Rastrigin问题
def Rastrigin01(x):
    label = []
    d = x.shape[1]
    #f = np.zeros(x.shape[0])
    cons = np.zeros((x.shape[0], 2))

    # 计算目标值
    s = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)
    f =  10 * d + s

    # 计算约束值与标签
    for i in range(x.shape[0]):
        cons[i, 0] = (5 - np.sum(x[i, :]))/d
        cons[i, 1] = np.sum(x[i, :int(d / 2)]) / int(d / 2) - np.sum(x[i, int(d / 2):]) / (d - int(d / 2))
        if (cons[i, 0] <= 0 and cons[i, 1] <= 0):
            label.append(0)
        else:
            label.append(1)
    return f,cons,np.array(label)

#非线性约束Rastrigin问题
def Rastrigin02(x):
    label = []
    d = x.shape[1]
    #f = np.zeros(x.shape[0])
    cons = np.zeros((x.shape[0], 2))

    # 计算目标值
    s = np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)
    f =  10 * d + s

    # 计算约束值与标签
    for i in range(x.shape[0]):
        cons[i, 0] = 10 * d - np.sum((x[i, :] + 1) ** 2)
        cons[i, 1] = 10 * d - np.sum((x[i, :] - 1) ** 2)

        if (cons[i, 0] <= 0 and cons[i, 1] <= 0):
            label.append(0)
        else:
            label.append(1)

    return f,cons,np.array(label)










