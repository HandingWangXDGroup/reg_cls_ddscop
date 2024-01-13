# -*- coding: utf-8 -*-
import RBFN
import multiprocessing as mp
import geatpy as ea
from smt.sampling_methods import LHS
from DE_Rand_Best_bin import DE_randTobest_1_bin
from realFunc import *
from BMProblem import allProblem
from sklearn.svm import SVR,SVC
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from imblearn.ensemble import BalancedBaggingClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

from pylab import mpl
#mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
#mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
import warnings
warnings.filterwarnings("ignore")



# 生成离线数据
def createData(xlimits ,num ,problem):
    sampling = LHS(xlimits=xlimits, criterion="cm")
    X = sampling(num)
    f,cons,l = problem(X)
    #print("测试集数据分布：可行样本个数：{0},不可行样本个数：{1},可行占比{2}".format((l == 0).sum(), (l == 1).sum(),(l == 0).sum()/num))
    return X ,f ,cons,l

def initial_model(problem,realFC,D,flag):

    """=================================种群设置=============================="""
    Encoding = 'RI'  # 编码方式RI
    NIND = 100 #5*D if 5*D < 100 else 100 # 种群规模

    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器

    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）

    SaSEGA = DE_randTobest_1_bin(problem, population,realFC,flag)  # 实例化一个算法模板对象
    SaSEGA.MAXGEN = 100 # 最大进化代数


    return SaSEGA



'''
实验测试：
1.真实对照（1次）
2.测试不同的代理模型性能（20次平均）
3.不同约束处理技术
'''

def figshow1(SaSEGA,prob_name, model_name,D):
    plt.figure(figsize=[16, 9])
    plt.title("Convergence of objective: "+model_name + " on "+str(D)+"-dimensional "+prob_name, fontsize=40)
    plt.plot(np.arange(100), SaSEGA.log['f'], c='red', label="Objective function")
    plt.scatter(np.arange(100), SaSEGA.log['SA'], c="pink", marker="v", label="Predicted function")
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel("Number of population iterations", fontsize=40)
    plt.ylabel("Objective value", fontsize=40)
    plt.legend(prop={"size": 35})
    plt.savefig("./imgs/objective_" + model_name + "_" + str(D)+"-dimensional_"+prob_name + ".eps", format='eps',bbox_inches='tight')
    plt.close()
    #plt.show()

    plt.figure(figsize=[16, 9])
    plt.title("Convergence of constraints: "+model_name + " on "+str(D)+"-dimensional "+prob_name, fontsize=40)
    plt.plot(np.arange(100), SaSEGA.log['cons1'], ls = "--",c='blue', label="Constraint function 1")
    plt.scatter(np.arange(100), SaSEGA.log['cons1SA'], c='skyblue', marker='P', label="Predicted constraint 1")
    plt.plot(np.arange(100), SaSEGA.log['cons2'], c='purple', label="Constraint function 2")
    plt.scatter(np.arange(100), SaSEGA.log['cons2SA'], c='orchid', marker='x', label="Predicted constraint 2")
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel("Number of population iterations", fontsize=40)
    plt.ylabel("Constraints value", fontsize=40)
    plt.legend(prop={"size": 35})
    plt.savefig("./imgs/constraints_" + model_name + "_" + str(D) + "-dimensional_" + prob_name + ".eps", format='eps',bbox_inches='tight')
    #plt.show()
    plt.close()

def figshow2(SaSEGA,prob_name,model_name,D):
    plt.figure(figsize=[16, 9])
    plt.title("Convergence of objective: "+model_name + " on " +" "+str(D)+"-dimensional "+prob_name, fontsize=40)
    plt.plot(np.arange(100), SaSEGA.log['f'], c='red', label="Objective function")
    plt.scatter(np.arange(100), SaSEGA.log['SA'], c="pink", marker="v", label="Predicted function")
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel("Number of population iterations", fontsize=40)
    plt.ylabel("Objective value", fontsize=40)
    plt.legend(prop={"size": 35})
    plt.savefig("./imgs/objective_" + model_name + "_" + str(D) + "-dimensional_" + prob_name + ".eps", format='eps',bbox_inches='tight')
    plt.close()
    #plt.show()

    plt.figure(figsize=[16, 9])
    plt.title("Convergence of constraints: " +model_name + " on " +" "+str(D)+"-dimensional "+prob_name, fontsize=40)
    plt.plot(np.arange(100), SaSEGA.log['cons1'],ls = "--", c='blue', label="Constraint function 1")
    plt.plot(np.arange(100), SaSEGA.log['cons2'], c='purple', label="Constraint function 2")
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.xlabel("Number of population iterations", fontsize=40)
    plt.ylabel("constraints value", fontsize=40)
    plt.legend(prop={"size": 35})
    plt.savefig("./imgs/constraints_" + model_name + "_" + str(D) + "-dimensional_" + prob_name + ".eps", format='eps',bbox_inches='tight')
    plt.close()
    #plt.show()

def test12(problem,realFC,D,flag,prob_name, model_name):
    SaSEGA = initial_model(problem, realFC, D, flag)
    lastpop, ndspop = SaSEGA.run()  # 执行算法模板，得到最优个体以及最后一代种群 #
    #nds_f, nds_cons, nds_l = realFC(ndspop[[0]].Phen)

    if flag == "reg":
        figshow1(SaSEGA, prob_name, model_name, D)
        pass
    else:
        figshow2(SaSEGA, prob_name, model_name, D)
        '''
        # 画置信度与目标值图
        nds_obj, _, nds_label = realFC(ndspop.Phen)
        if flag == "svc":
            nds_cv = problem.Scon.decision_function(ndspop.Phen)
        else:
            nds_cv = problem.Scon.predict_proba(ndspop.Phen)[:, [1]]

        plt.figure(figsize=[8, 6])
        plt.title( model_name+" on "+str(D) + "-dimensional "+prob_name+":Non-dominated solutions", fontsize=15)
        plt.scatter(nds_cv[nds_label == 0], nds_obj[nds_label == 0], c='g', label="feasible solution")
        plt.scatter(nds_cv[nds_label == 1], nds_obj[nds_label == 1], c='r', marker='x', label="unfeasible solution")
        if flag == "svc":
            plt.xlabel("Distance from decision surface", fontsize=15)
        else:
            plt.xlabel("Infeasibility Confidence", fontsize=15)
        plt.ylabel("Objective value", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.legend(prop={"size": 15})
        plt.savefig("./imgs/nds_" + model_name + "_" + str(D) + "-dimensional_" + prob_name + ".eps",
                    format='eps')
        '''


def test11(problem,realFC,D,flag):

    SaSEGA = initial_model(problem, realFC,D,flag)
    lastpop, ndspop = SaSEGA.run()  # 执行算法模板，得到最优个体以及最后一代种群 #
    nds_f, nds_cons, nds_l = realFC(lastpop[[0]].Phen)

    return nds_f,nds_cons,nds_l




def test30(F, D,prob_name):
    count = np.zeros(6)
    feasibleBestSolution = [[]]*6

    flags = ["reg", "reg", "reg", "gau", "svc", "esvc"]
    model_names = ["Linear Regression", "SVM Regression", "RBFN", "Naive Bayesian Classifier", "SVM Classifier",
                   "Ensemble SVMs Classifier"]

    for n in range(30):

        xlimits = np.array([[-5.12, 5.12]] * D)
        X, y, cons, l = createData(xlimits, 11 * D, F)

        #目标代理
        Objreg = RBFN.RBFN(num_neurons=int(np.sqrt(11 * D)), kernel='gaussian')
        Objreg.fit(X, y)

        #线性回归
        Conreg1 = [LinearRegression(),
                   LinearRegression()]
        Conreg1[0].fit(X, cons[:, 0])
        Conreg1[1].fit(X, cons[:, 1])

        # svr 支持向量回归器(网格搜索调参)
        parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]}
        gs1 = GridSearchCV(SVR(), parameters, cv=5,
                           scoring='r2')
        gs2 = GridSearchCV(SVR(), parameters, cv=5,
                           scoring='r2')
        gs1.fit(X, cons[:, 0])
        gs2.fit(X, cons[:, 1])

        Conreg2 = [gs1.best_estimator_,
                   gs2.best_estimator_]
        Conreg2[0].fit(X, cons[:, 0])
        Conreg2[1].fit(X, cons[:, 1])

        # RBFN 约束值回归器
        Conreg3 = [RBFN.RBFN(num_neurons=int(np.sqrt(11 * D)), kernel='gaussian'),
                   RBFN.RBFN(num_neurons=int(np.sqrt(11 * D)), kernel='gaussian')]
        Conreg3[0].fit(X, cons[:, 0])
        Conreg3[1].fit(X, cons[:, 1])

        '''
        三种约束分类器对比
        '''
        cls1 = GaussianNB()  # gau
        cls1.fit(X, l)

        cls2 = SVC(kernel="rbf", C=100)  # svc
        cls2.fit(X, l)

        cls3 = BalancedBaggingClassifier(base_estimator=SVC(kernel="rbf", C=100, probability=True),
                                         n_estimators=10)

        cls3.fit(X, l)

        problst = [ allProblem(D, Objreg, Conreg1, flag="reg"),
                    allProblem(D, Objreg, Conreg2, flag="reg"),
                    allProblem(D, Objreg, Conreg3, flag="reg"),
                    allProblem(D, Objreg, cls1, flag="gau"),
                    allProblem(D, Objreg, cls2, flag="svc"),
                    allProblem(D, Objreg, cls3, flag="esvc")
                ]


        for i in range(6):
            nds_f, nds_cons, nds_l = test11(problst[i], F, D, flags[i])
            if (nds_l[0] == 0):
                # print("最优个体函数值：{},约束值：{},标签：{}。评价次数：{}，时间已过 {} 秒".format(nds_f, nds_cons, nds_l,SaSEGA.evalsNum, SaSEGA.passTime))
                count[i] += 1
                feasibleBestSolution[i] =feasibleBestSolution[i] + [nds_f[0]]
    for i in range(6):
        if count[i] == 0:
            print(model_names[i] + " 在" + str(D) + "维" + prob_name + "上，" + "可行率为：%f" % (count[i] / 30.0))
        else:
            print(model_names[i] + " 在" + str(D) + "维" + prob_name+ "上，" + "可行最优解的平均值为：%s,标准差为：%s,可行率为：%f" % (
                np.mean(feasibleBestSolution[i]), np.std(feasibleBestSolution[i]), count[i] / 30.0))



def plottest2(D,F,prob_name):
    flags = ["reg", "reg", "reg", "gau", "svc", "esvc"]
    model_names = ["LR", "SVR", "RBFNs", "GNB", "SVM",
                   "ESVMs"]
    xlimits = np.array([[-5.12, 5.12]] * D)
    X, y, cons, l = createData(xlimits, 11 * D, F)


    # 目标代理
    Objreg = RBFN.RBFN(num_neurons=int(np.sqrt(11 * D)), kernel='gaussian')
    Objreg.fit(X, y)

    # 线性回归
    Conreg1 = [LinearRegression(),
               LinearRegression()]
    Conreg1[0].fit(X, cons[:, 0])
    Conreg1[1].fit(X, cons[:, 1])

    # svr 支持向量回归器(网格搜索调参)
    parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10, 100]}
    gs1 = GridSearchCV(SVR(), parameters, cv=5,
                       scoring='r2')
    gs2 = GridSearchCV(SVR(), parameters, cv=5,
                       scoring='r2')
    gs1.fit(X, cons[:, 0])
    gs2.fit(X, cons[:, 1])

    Conreg2 = [gs1.best_estimator_,
               gs2.best_estimator_]
    Conreg2[0].fit(X, cons[:, 0])
    Conreg2[1].fit(X, cons[:, 1])

    # RBFN 约束值回归器
    Conreg3 = [RBFN.RBFN(num_neurons=int(np.sqrt(11 * D)), kernel='gaussian'),
               RBFN.RBFN(num_neurons=int(np.sqrt(11 * D)), kernel='gaussian')]
    Conreg3[0].fit(X, cons[:, 0])
    Conreg3[1].fit(X, cons[:, 1])

    '''
    三种约束分类器对比
    '''
    cls1 = GaussianNB()  # gau
    cls1.fit(X, l)

    cls2 = SVC(kernel="rbf", C=100)  # svc
    cls2.fit(X, l)

    cls3 = BalancedBaggingClassifier(base_estimator=SVC(kernel="rbf", C=100, probability=True),
                                     n_estimators=10)

    cls3.fit(X, l)

    problst = [allProblem(D, Objreg, Conreg1, flag="reg"),
               allProblem(D, Objreg, Conreg2, flag="reg"),
               allProblem(D, Objreg, Conreg3, flag="reg"),
               allProblem(D, Objreg, cls1, flag="gau"),
               allProblem(D, Objreg, cls2, flag="svc"),
               allProblem(D, Objreg, cls3, flag="esvc")
               ]

    for i in range(6):
        test12(problst[i], F, D, flags[i], prob_name, model_names[i])

if __name__ == '__main__':

    F_lst = [Ellipsoid01, Ellipsoid02, Rastrigin01, Rastrigin02]
    prob_name = ["F1", "F2", "F3", "F4"]
    for i in [0]:
        for D in [10]:
            print("实验： " + prob_name[i] + "问题_" + str(D) + "维")
            #plottest2(D, F_lst[i], prob_name[i])
            test30(F_lst[i], D, prob_name[i])




















