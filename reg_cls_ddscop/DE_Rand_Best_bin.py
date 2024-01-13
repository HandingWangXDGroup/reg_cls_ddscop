# -*- coding: utf-8 -*-
import geatpy as ea  # 导入geatpy库
import numpy as np
import random
from soeaAlgorithm import SoeaAlgorithm
import itertools



class DE_randTobest_1_bin(SoeaAlgorithm):

    def __init__(self, problem, population,realFC,flag):
        SoeaAlgorithm.__init__(self, problem, population,realFC,flag)  # 先调用父类构造方法
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'DE/rand/1/bin'
        if population.Encoding == 'RI':
            self.mutOper = ea.Mutde(F=0.5)  # 生成差分变异算子对象
            self.recOper = ea.Xovbd(XOVR=0.8, Half_N=True)  # 生成二项式分布交叉算子对象，这里的XOVR即为DE中的Cr
        else:
            raise RuntimeError('编码方式必须为''RI''.')
        self.flag = flag


    def cal_cv(self,CV):
        cv = np.maximum(CV,0)
        cvs = cv[:,0]/np.max(cv[:,0]) + cv[:,1]/np.max(cv[:,1])
        return cvs




    # p支配q，返回1
    def is_dominated1(self, o1, o2, c1, c2):
        if (o1 > o2 or c1 > c2):
            return 0
        elif (o1 == o2 and c1 == c2):
            return 0
        else:
            return 1
    def is_dominated2(self, o1, o2, c1_1, c1_2,c2_1,c2_2):
        if (o1 > o2 or c1_1 > c1_2 or c2_1 > c2_2):
            return 0
        elif (o1 == o2 and c1_1 == c1_2 and c2_1 == c2_2):
            return 0
        else:
            return 1

    def fast_nds(self, pop):
        sp_lst = []
        np_lst = []
        rank_lst = np.zeros(pop.sizes)  # 保存每个个体所处支配层

        F1 = []
        # 寻找pareto第一级个体
        for i in range(pop.sizes):
            n_p = 0
            sp = []

            for j in range(pop.sizes):
                if (i != j):
                    if self.flag == "reg":
                        cond1 = self.is_dominated2(pop.ObjV[i], pop.ObjV[j], pop.CV[i,0],pop.CV[j,0],pop.CV[i,1],pop.CV[j,1])
                        cond2 = self.is_dominated2(pop.ObjV[j], pop.ObjV[i], pop.CV[j,0],pop.CV[i,0],pop.CV[j,1],pop.CV[i,1])
                    else:
                        cond1 = self.is_dominated1(pop.ObjV[i], pop.ObjV[j], pop.CV[i], pop.CV[j])
                        cond2 = self.is_dominated1(pop.ObjV[j], pop.ObjV[i], pop.CV[j], pop.CV[i])

                    if (cond1):
                        sp.append(j)
                    elif (cond2):
                        n_p += 1

            if (n_p == 0):
                rank_lst[i] = 1  # np为0，个体为pareto第一级
                F1.append(i)
            np_lst.append(n_p)
            sp_lst.append(sp)
        return F1

    def CHT1(self, tempPop, population):
        # 先找可行解，并按目标值大小排序
        feasible = []
        for i in range(tempPop.sizes):
            if (tempPop.CV[i, 0] <= 0 and tempPop.CV[i, 1] <= 0):
                feasible.append(i)

        feapop = tempPop[feasible]
        Nfpop = tempPop[list(set(range(tempPop.sizes)).difference(set(feasible)))]

        # 预测可行个数大于种群个数
        if len(feasible) >= population.sizes:
            if (np.random.rand() < 0.445):
                population = feapop[np.argsort(feapop.ObjV, axis=0)[:population.sizes]]
            else:
                population = feapop[random.sample(range(feapop.sizes), population.sizes)]

        else:
            cvs = self.cal_cv(Nfpop.CV)
            population = feapop + Nfpop[[np.argsort(cvs)[:(population.sizes - feapop.sizes)]]]

        population.FitnV = (np.max(population.ObjV) - population.ObjV + 1) / (
                1 + np.max(population.ObjV) - np.min(population.ObjV))
        return population

    def CHT2(self, tempPop, population):
        if self.flag == "svc":
            feasible = np.where(tempPop.CV <= 0)[0]
        else:
            feasible = np.where(tempPop.CV <= 0.15)[0]

        feapop = tempPop[feasible]
        Nfpop = tempPop[list(set(range(tempPop.sizes)).difference(set(feasible)))]

        # 预测可行个数大于种群个数
        if len(feasible) >= population.sizes:
            if (np.random.rand() < 0.445):
                population = feapop[np.argsort(feapop.ObjV, axis=0)[:population.sizes]]
            else:
                population = feapop[random.sample(range(feapop.sizes), population.sizes)]
                #population = feapop[np.argsort(feapop.CV)[:population.sizes]]  #random.sample(range(feapop.sizes), population.sizes)

        else:
            population = feapop + Nfpop[[np.argsort(Nfpop.CV)[:(population.sizes - feapop.sizes)]]]

        population.FitnV = (np.max(population.ObjV) - population.ObjV + 1) / (
                1 + np.max(population.ObjV) - np.min(population.ObjV))

        return population


    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）

        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法模板的一些动态参数

        # ===========================准备进化============================
        population.initChrom(NIND)  # 初始化种群染色体矩阵
        self.call_aimFunc(population)  # 计算种群的目标函数值

        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查，故应确保prophetPop是一个种群类且拥有合法的Chrom、ObjV、Phen等属性）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群

        population.FitnV = (np.max(population.ObjV) - population.ObjV + 1) / ( 1 + np.max(population.ObjV) - np.min(population.ObjV))  # 计算适应度

        # 非支配保存
        nds_pop = population[self.fast_nds(population)]

        # ===========================开始进化============================
        while self.terminated(population) == False:

            # 基向量的选择方式，采用随机补偿选择与最优
            if self.currentGen <= 50:
                selFunc = 'rcs'
            else:
                selFunc = 'ecs'

            r0 = ea.selecting(selFunc, population.FitnV, NIND)  # 得到基向量索引

            experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            experimentPop.Chrom = self.mutOper.do(population.Encoding, population.Chrom, population.Field, [r0])  # 变异
            experimentPop.Chrom = self.recOper.do(np.vstack([population.Chrom, experimentPop.Chrom]))  # 重组


            self.call_aimFunc(experimentPop)  # 计算目标函数值


            tempPop = population + experimentPop  # 临时合并，80个个体


            if self.flag == "reg":
                population = self.CHT1(tempPop, population)
            else:
                population = self.CHT2(tempPop, population)

            #population = population[np.argsort(population.ObjV,axis=0)]
            #poptemp = nds_pop + population
            #nds_pop= poptemp[self.fast_nds(poptemp)]
            #if nds_pop.sizes > population.sizes:
           #     nds_pop = nds_pop[random.sample(range(nds_pop.sizes), population.sizes)]

        return self.finishing(population),nds_pop  # 调用finishing完成后续工作并返回结果

