#coding:utf8

import numpy as np
import pandas as pd
import tqdm
import tryMultiList as tml

def getTimeItemData():
    '''获取 时间-商品 数据'''
    trainDf=pd.read_csv("data/Antai_AE_round1_train_20190626.csv").sort_values(by=["buyer_admin_id","irank"])
    trainDf["create_order_time"]=trainDf["create_order_time"].apply(lambda x:x.split(" ")[0])
    timeList=list(set(np.array(trainDf["create_order_time"]).tolist()))
    timeItemList=[[timeItem,",".join([str(intItem) for intItem in np.array(trainDf.loc[trainDf["create_order_time"]==timeItem,"item_id"]).tolist()])]\
                   for timeItem in tqdm.tqdm(timeList)]
    timeItemDf=pd.DataFrame(np.array(timeItemList),columns=["create_order_time","item_id"])
    timeItemDf.to_csv("data/timeItemData.csv")
    print("finished")
    
def getItemBuyerDataSingleTh(itemList,argList):
    '''单线程获取 商品-用户 数据，用于装载多线程'''
    trainDf=argList[0]
    return [[itemBuyer,",".join([str(intItem) for intItem in np.array(trainDf.loc[trainDf["item_id"]==itemBuyer,"buyer_admin_id"]).tolist()])] for itemBuyer in itemList]

def getItemBuyerData():  
    '''获取 商品-用户 数据'''
    trainDf = pd.read_csv("data/Antai_AE_round1_train_20190626.csv").sort_values(by=["buyer_admin_id","irank"])[:100000]
    itemList=list(set(np.array(trainDf["item_id"]).tolist()))
    itemBuyerList=tml.multiListProcess(getItemBuyerDataSingleTh,[trainDf],itemList)
    itemBuyerDf=pd.DataFrame(np.array(itemBuyerList),columns=["item_id","buyer_admin_id"])
    itemBuyerDf.to_csv("data/itemBuyerData.csv")
    print("finished")

def getBuyerItemDataSingleTh(buyerList,argList):
    '''单线程获取 用户-商品 数据，用于装载多线程'''
    trainDf=argList[0]
    return [[buyerItem,",".join([str(intItem) for intItem in np.array(trainDf.loc[trainDf["buyer_admin_id"]==buyerItem,"item_id"]).tolist()])] for buyerItem in buyerList]
    
def getBuyerItemData():
    '''获取 用户-商品 数据'''
    trainDf = pd.read_csv("data/Antai_AE_round1_train_20190626.csv").sort_values(by=["buyer_admin_id","irank"])[:100000]
    buyerList=list(set(np.array(trainDf["buyer_admin_id"]).tolist()))
    buyerItemList=tml.multiListProcess(getBuyerItemDataSingleTh,[trainDf],buyerList)
    buyerItemDf=pd.DataFrame(np.array(buyerItemList),columns=["buyer_admin_id","item_id"])
    buyerItemDf.to_csv("data/buyerItemData.csv")
    print("finished")

if __name__=="__main__":
    # getTimeItemData()
    getItemBuyerData()
    getBuyerItemData()