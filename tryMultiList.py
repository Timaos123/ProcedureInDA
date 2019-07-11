#coding:utf8
import copy
from multiprocessing import Pool,cpu_count
import matplotlib.pyplot as plt
import time
import tqdm
import A1_preprocessingForML as A1
import pandas as pd
import numpy as np

def multiListProcess(fun,argList,myList,slice_num=4):
    '''
    inputs:
    fun: the name of the function
    argList: the list of arguments
    myList: the original list
    slice_num: the number of the slices after myList is cut up
    ======================================
    return:
    the list after processed
    '''
    p=Pool(8)
    sliceResultMulList=[]
    for i in tqdm.tqdm(range(slice_num)):
        sliceResult=p.apply_async(fun,args=[myList[int(len(myList)/slice_num)*i:int(len(myList)/slice_num)*(i+1)],argList])
        sliceResultMulList+=sliceResult.get()
    p.close()
    p.join()
    return sliceResultMulList

def normal(xList,args):
    import numpy as np
    return [[i for i in range(5000)] for x in xList]

if __name__=="__main__":
    # nv1TimeList=[]
    # nv2TimeList=[]
    # for rangeNO in tqdm.tqdm(range(10000,50000,10000)):
    #     xList=[i for i in range(rangeNO)]
    #     mean=5/2
    #     std=5/4

    #     start=time.time()
    #     nv1=normal(xList,(mean,std))
    #     end=time.time()
    #     nv1Time=end-start
    #     nv1TimeList.append(nv1Time)

    #     start=time.time()
    #     nv2=multiListProcess(normal,[mean,std],xList,slice_num=1000)
    #     end=time.time()
    #     nv2Time=end-start
    #     nv2TimeList.append(nv2Time)
    timeList=[]
    start=time.time()
    A1.getBuyerItemData()
    end=time.time()
    timeList.append(end-start)

    # start=time.time()
    # trainDf = pd.read_csv("data/Antai_AE_round1_train_20190626.csv")[:50000]
    # itemList=list(set(np.array(trainDf["item_id"]).tolist()))
    # A1.getItemBuyerDataSingleTh(itemList,[trainDf])
    # end=time.time()
    # timeList.append(end-start)
    # plt.plot(timeList)
    # plt1,=plt.plot(nv1TimeList)
    # plt2,=plt.plot(nv2TimeList)
    # plt.legend([plt1,plt2],["plt1","plt2"],loc = 'upper right')
    # plt.show()