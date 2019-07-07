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


    
if __name__=="__main__":
    getTimeItemData()


