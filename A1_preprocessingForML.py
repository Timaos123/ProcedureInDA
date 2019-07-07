#coding:utf8

import pandas as pd
import numpy as np
import tqdm

def getTimeItemData():
    '''获取 时间-商品 数据'''
    trainDf=pd.read_csv("data/Antai_AE_round1_train_20190626.csv")
    trainDf["create_order_time"]=trainDf["create_order_time"].apply(lambda x:x.split(" ")[0])
    timeList=list(set(np.array(trainDf["create_order_time"]).tolist()))
    timeItemList=[[timeItem,",".join([str(intItem) for intItem in np.array(trainDf.loc[trainDf["create_order_time"]==timeItem,"item_id"]).tolist()])] for timeItem in tqdm.tqdm(timeList)]
    timeItemDf=pd.DataFrame(np.array(timeItemList),columns=["create_order_time","item_id"])
    timeItemDf.to_csv("data/timeItemData.csv")
    print("finished")

 def getItemUserData():  
    '''获取 商品-用户 数据'''
    trainDf = pd.read_csv("data/Antai_AE_round1_train_20190626.csv")
    itemList=list(set(np.array(trainDf["item_id"]).tolist()))
    itemBuyerList=[[timeItem,",".join([str(intItem) for intItem in np.array(trainDf.loc[trainDf["item_id"]==timeItem,"buyer_admin_id"]).tolist()])] for timeItem in tqdm.tqdm(itemList)]
    itemBuyerDf=pd.DataFrame(np.array(itemBuyerList),columns=["item_id","buyer_admin_id"])
    itemBuyerDf.to_csv("data/itemBuyerData.csv")
    print("finished")  
    
if __name__=="__main__":
    getTimeItemData()
    getItemUserData()

