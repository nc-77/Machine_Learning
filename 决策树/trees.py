# -*- coding: utf-8 -*-
# In[0]
# 调库和全局变量
from math import log
from collections import Counter
import numpy as np
if __name__=='main':
    dataset=np.array([[1,1,1],
             [1,1,1],
             [1,0,0],
             [0,1,0],
             [0,1,0]])
    features=["no surfacing","flippers"]
# In[1] 
# calcEnt 计算给定数据集的香农熵
# 参数:
#    dataset 原始数据集(带标签)
def calcEnt(dataset):
    labels=dataset[:,-1]
    labels_cnt=Counter(labels)
    Ent=0
    for label_cnt in labels_cnt.values():
        prob=label_cnt/len(labels)
        Ent+=-log(prob,2)*prob
    return Ent
if __name__=='main':
    Ent=calcEnt(dataset)
# In[2]
# splitDataSet 划分数据集
# 参数: 
#    dataset:原始数据集
#    axis:划分数据集的特征
#    value:需要返回的特征的值
def splitDataSet(dataset,axis,value):
    return dataset[dataset[:,axis]==value]
if __name__=='main':
    ret=splitDataSet(dataset,dataset.shape[1]-1,'yes')
# In[3]
# chooseBestFeature 选择最好的数据划分方式
# 参数:
#    dataset:原始数据集
# 返回信息增益最大的特征
def chooseBestFeature(dataset):
    oldEnt=calcEnt(dataset)
    BestinfoGain=bestfeature=0
    n=dataset.shape[1]-1
    for i in range(n):
        newEnt=0
        values=set(dataset[:,i])
        for value in values:
            subDataSet=splitDataSet(dataset, i, value)
            newEnt+=calcEnt(subDataSet)*len(subDataSet)/len(dataset)
        infoGain=oldEnt-newEnt
        if(infoGain>BestinfoGain):
            BestinfoGain=infoGain
            bestfeature=i
    return bestfeature
if __name__=='main':
    bestfeature=chooseBestFeature(dataset)
# In[4]
# createTree 创建决策树
# 参数: 
#    dataset:原始数据集
def createTree(dataset,features):
    if len(dataset[0])==1:
        return Counter(dataset)[0].key()
    if len(Counter(dataset[:,-1]))==1:
        return dataset[0][-1]
    index=chooseBestFeature(dataset)
    bestfeat=features[index]
    mytree={bestfeat:{}}
    featvalue=set(dataset[:,index])
    for value in featvalue:
        mytree[bestfeat][value]=createTree(splitDataSet(dataset,index,value), features)
    return mytree
if __name__=='main':
    print(createTree(dataset, features))
# In[5]
# classify 根据决策树进行分类
# 参数:
#    inX:待分类样本
#    Tree:训练好的决策树
    
def classify(inX,Tree,features):
    root=list(Tree.keys())[0]
    index=features.index(root)
    for key in Tree[root].keys():
        if(key==inX[index]):
            if(type(Tree[root][key]).__name__=='dict'):
                class_label=classify(inX,Tree[root][key],features)
            else:
                class_label=Tree[root][key]
    return class_label
if __name__=='main':
    mytree=createTree(dataset,features)
    inX=[0,0]
    print(classify(inX, mytree))