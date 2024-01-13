# -*- coding: utf-8 -*-

import geatpy as ea
from realFunc import *

class allProblem(ea.Problem):  # 继承Problem父类
    def __init__(self,Dim,Objreg,Scon,flag):

        name = "Ellipsoid01"+str(Dim)  # 初始化name（函数名称，可以随意设置）

        self.Scon = Scon
        self.Objreg = Objreg
        self.flag = flag
        M = 1  # 初始化M（目标维数）
        maxormins = [1]        # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）# 初始化Dim（决策变量维数）
        varTypes = [0] * Dim   # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [-5.12]*Dim       # 决策变量下界
        ub = [5.12]*Dim        # 决策变量上界
        lbin = [1] * Dim       # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim       # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def aimFunc(self, pop):  # 目标函数

        x = pop.Phen  # 得到决策变量矩阵

        pop.ObjV = self.Objreg.predict(x).reshape(-1, 1)

        if self.flag == "reg":
            pop.CV = np.column_stack([self.Scon[0].predict(x), self.Scon[1].predict(x)])
        elif self.flag == "svc":
            pop.CV = self.Scon.decision_function(x).reshape(-1, 1)  # 到决策平面的距离
        else:
            pop.CV = self.Scon.predict_proba(x)[:, [1]]





'''
        #self.atype = atype
        #self.Concls = Concls
        if self.atype == "DDEA_cls":         #分类器对约束建模
            pop.ObjV = self.Objreg.predict(x).reshape(-1, 1)
            pop.CV = self.Concls.predict_proba(x)[:, [1]]  # 约束违反值

        elif self.atype == "DDEA_reg":       #回归器对约束建模

class Rastrigin(ea.Problem):  # 继承Problem父类
    def __init__(self,Dim):
        name = "Rastrigin01"+str(Dim)  # 初始化name（函数名称，可以随意设置）

        M = 1  # 初始化M（目标维数）
        maxormins = [1]        # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）# 初始化Dim（决策变量维数）
        varTypes = [0] * Dim   # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [-5.12]*Dim       # 决策变量下界
        ub = [5.12]*Dim        # 决策变量上界
        lbin = [1] * Dim       # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim       # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def aimFunc(self, pop):  # 目标函数
        x = pop.Phen  # 得到决策变量矩阵

        # 约束条件
        #pop.CV = self.ConF(x).reshape(-1,1)
        #pop.CV = self.cls.predict(x).reshape(-1,1)       #按可行结果0或1
        pop.CV = self.cls.predict_proba(x)[:, [1]]         #约束违反值

        #pop.ObjV = F2(x).reshape(-1, 1)
        pop.ObjV = self.reg.predict(x).reshape(-1,1)
        #pop.ObjV = self.reg._predict_values(x).reshape(-1,1)
'''
