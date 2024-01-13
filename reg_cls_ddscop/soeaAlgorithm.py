# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea
import warnings
import time


class Algorithm:


    def __init__(self):

        """
        描述:
            构造函数。

        """
        self.name = 'Algorithm'
        self.problem = None
        self.population = None
        self.MAXGEN = None
        self.currentGen = None
        self.MAXTIME = None
        self.timeSlot = None
        self.passTime = None
        self.MAXEVALS = None
        self.evalsNum = None
        self.MAXSIZE = None
        self.log = None


    def initialization(self):
        pass

    def run(self, pop):
        pass

    def logging(self, pop):
        pass

    def stat(self, pop):
        pass

    def terminated(self, pop):
        pass

    def finishing(self, pop):
        pass

    def check(self, pop):
        # 检测数据非法值
        if np.any(np.isnan(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are NAN, please check the calculation of ObjV.(ObjV的部分元素为NAN，请检查目标函数的计算。)",
                RuntimeWarning)
        elif np.any(np.isinf(pop.ObjV)):
            warnings.warn(
                "Warning: Some elements of ObjV are Inf, please check the calculation of ObjV.(ObjV的部分元素为Inf，请检查目标函数的计算。)",
                RuntimeWarning)
        if pop.CV is not None:
            if np.any(np.isnan(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are NAN, please check the calculation of CV.(CV的部分元素为NAN，请检查CV的计算。)",
                    RuntimeWarning)
            elif np.any(np.isinf(pop.CV)):
                warnings.warn(
                    "Warning: Some elements of CV are Inf, please check the calculation of CV.(CV的部分元素为Inf，请检查CV的计算。)",
                    RuntimeWarning)

    def call_aimFunc(self, pop):

        pop.Phen = pop.decoding()  # 染色体解码
        if self.problem is None:
            raise RuntimeError('error: problem has not been initialized. (算法模板中的问题对象未被初始化。)')
        self.problem.aimFunc(pop)  # 调用问题类的aimFunc()
        self.evalsNum = self.evalsNum + pop.sizes if self.evalsNum is not None else pop.sizes  # 更新评价次数
        # 格式检查
        if not isinstance(pop.ObjV, np.ndarray) or pop.ObjV.ndim != 2 or pop.ObjV.shape[0] != pop.sizes or \
                pop.ObjV.shape[1] != self.problem.M:
            raise RuntimeError('error: ObjV is illegal. (目标函数值矩阵ObjV的数据格式不合法，请检查目标函数的计算。)')
        if pop.CV is not None:
            if not isinstance(pop.CV, np.ndarray) or pop.CV.ndim != 2 or pop.CV.shape[0] != pop.sizes:
                raise RuntimeError('error: CV is illegal. (违反约束程度矩阵CV的数据格式不合法，请检查CV的计算。)')


class SoeaAlgorithm(Algorithm):  # 单目标优化算法模板父类

    def __init__(self, problem, population,realFC,flag):
        super().__init__()  # 先调用父类构造函数
        self.problem = problem
        self.population = population
        self.realFC = realFC
        self.flag = flag
        self.trappedValue = 0  # 默认设置trappedValue的值为0
        self.maxTrappedCount = 1000  # 默认设置maxTrappedCount的值为1000

    def initialization(self):


        self.passTime = 0  # 初始化passTime
        self.trappedCount = 0  # 初始化“进化停滞”计数器
        self.currentGen = 0  # 初始为第0代
        self.evalsNum = 0  # 初始化评价次数为0
        self.log = {'gen': [], 'eval': []}   # 初始化log

        # 开始计时
        self.timeSlot = time.time()

    def logging(self, pop):
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
        if len(self.log['gen']) == 0:  # 初始化log的各个键值
            self.log['SA'] = []
            self.log['f'] = []
            self.log['cons1'] = []
            self.log['cons2'] = []


            self.log['cons1SA'] = []
            self.log['cons2SA'] = []


        self.log['gen'].append(self.currentGen)
        self.log['eval'].append(self.evalsNum)  # 记录评价次数

        f,cons,l = self.realFC(pop.decoding())
        arg = np.where(l==0)
        #print(f[arg], cons[arg])
        #exit(0)


        self.log['SA'].append(np.mean(pop.ObjV[arg]))
        self.log['f'].append(np.mean(f[arg]))
        self.log['cons1'].append(np.mean(cons[arg][:,0]))
        self.log['cons2'].append(np.mean(cons[arg][:,1]))
        if self.flag == "reg":
            self.log['cons1SA'].append(np.mean(pop.CV[arg][:, 0]))
            self.log['cons2SA'].append(np.mean(pop.CV[arg][:, 1]))

        self.timeSlot = time.time()  # 更新时间戳



    def terminated(self, pop):

        self.check(pop)  # 检查种群对象的关键属性是否有误
        self.logging(pop)  # 记录日志
        self.passTime += time.time() - self.timeSlot  # 更新耗时
        self.timeSlot = time.time()  # 更新时间戳
        # 判断是否终止进化，由于代数是从0数起，因此在比较currentGen和MAXGEN时需要对currentGen加1
        if (
                self.MAXTIME is not None and self.passTime >= self.MAXTIME) or self.currentGen + 1 >= self.MAXGEN or self.trappedCount >= self.maxTrappedCount:
            return True
        else:
            self.currentGen += 1  # 进化代数+1
            return False

    def finishing(self, pop):
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
        # 返回最后一代种群
        return  pop