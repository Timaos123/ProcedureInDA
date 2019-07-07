# ProcedureInDA 使用说明

数据集地址：[https://tianchi.aliyun.com/competition/entrance/231718/information]

## 1.文件架构
大致流程分布包括：A（数据描述及预处理）、B（模型训练）、C（模型预测）[、D（模型整合）]
目前所拥有的文件包括：

data：存储数据（不放到github上）

log：深度学习日志数据

model：模型数据（训练后存储）

A0_originalDataDes.py：原始数据描述

A1_preprocessingForML.py：传统机器学习预处理（基于矩阵的数据结构）

A2_preprocessingForDL.py：深度学习预处理（基于张量的数据结构）

B0_trainML.py：训练传统机器学习

B1_trainDL.py：训练深度学习

C0_predictWithML.py：传统机器学习预测

C1_predictWithDL.py：深度学习预测	

可在每一栏后新添自己相关新文件，并附带序号（eg.在B1后新添B2_trainSelf.py）

## 2.命名方式（皆单数形式命名）

### 变量命名（驼峰命名）：

合成词汇“首个字母小写的单词+首字母大写的单词”（如：“开头时间”变量命名：startTime=time.time()）

特殊情况：

  同一容器进行多次数据类型变换时标识其容器类型（如：myDataDf=pd.read_csv("...");myDataArr=np.array(myDataDf);myDataList=myDataArr.tolist()）
  
  对某容器内容进行遍历for循环内个体变量以Item结尾（如遍历停止词列表[stopWordItem for stopWordItem in stopWordList]）
  
### 文件命名：

#### py文件：
  
>流程py文件：
  
>>流程字母（A（数据描述及预处理）、B（模型训练）、C（模型预测）[、D（模型整合）]）流程子步骤编号_子步骤名称（驼峰）（eg.C1_predictWithDL.py）
  
>工具py文件：

>>驼峰命名，try开头（eg.trySQL.py）

#### 生成数据文件

>data文件夹内文件

>>pkl文件：标注pkl展开后的数据类型（eg.userItemDict.pkl）

>>训练测试集相关文件：train/test驼峰命名（eg.trainMLMatrix.pkl）

>>传统机器学习/深度学习相关文件：ML/DL驼峰命名（eg.trainMLMatrix.pkl）

>model文件夹内文件

>>模型名称Model（eg.SVMModel.h5）

## 3.代码架构

尽量把所有固定过程都放到子函数中

````
#coding:utf8

import pandas as pd
import numpy as np
import tqdm

def getTimeItemData():
    '''获取 时间-商品 数据'''#子函数有注释
    trainDf=pd.read_csv("data/Antai_AE_round1_train_20190626.csv")
    trainDf["create_order_time"]=trainDf["create_order_time"].apply(lambda x:x.split(" ")[0])
    timeList=list(set(np.array(trainDf["create_order_time"]).tolist()))
    timeItemList=[[timeItem,",".join([str(intItem) for intItem in np.array(trainDf.loc[trainDf["create_order_time"]==timeItem,"item_id"]).tolist()])] for timeItem in timeList]
    timeItemDf=pd.DataFrame(np.array(timeItemList),columns=["create_order_time","item_id"])
    timeItemDf.to_csv("data/timeItemData.csv")
    print("finished")

if __name__=="__main__":#要有主函数
    getTimeItemData()
````
