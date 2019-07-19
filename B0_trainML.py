#coding:utf8

import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.externals import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import KFold
import tqdm
import tryMultiList as tml
import scipy as sc
from numpy.linalg import svd

class CF:
    def __init__(self,method="svd",isTfidf=False,**args):
        '''
        if method==svd (default): 
            set keptInfo=[0,1] to set the kept information in the vectors
        '''
        self.method=method
        self.counter=CountVectorizer()
        self.tfidfTransformer=TfidfTransformer()
        self.isTfidf=isTfidf
        self.trainedMatModels=False

        self.buildModel(args)

    def buildModel(self,args):
        '''
        vectorization models
        '''
        if self.method=="svd":
            self.keptInfo=args["keptInfo"]

    def transformToMat(self,myList):
        '''
        transform data into matrix
        '''
        if self.method in ["svd"]:
            if self.trainedMatModels==False:#if the matrix models are untrained
                myMat=self.counter.fit_transform(myList)
                if self.isTfidf==True:
                    myMat=self.tfidfTransformer.fit_transform(myMat)
                self.trainedMatModels=True
            else:
                myMat=self.counter.transform(myList)
                if self.isTfidf==True:
                    myMat=self.tfidfTransformer.transform(myMat)
                self.trainedMatModels=True
        return myMat
    
    def train(self,xArr,yArr):
        '''
        training models
        '''
        if self.method=="svd":
            xList=xArr.tolist()
            yList=yArr.tolist()
            myList=[" ".join(xList[i].split(" ")+[yList[i]]) for i in range(len(xList))]
            myMat=self.transformToMat(myList)
            svdMatList=svd(myMat.A)
            infoMat=svdMatList[1]/np.sum(svdMatList[1])
            tempInfo=0
            keptDim=1
            for i in range(infoMat.shape[0]):
                keptDim+=1
                tempInfo=np.sum(infoMat[:i+1])
                if tempInfo>self.keptInfo:
                    break
            self.keptInfo=tempInfo
            self.keptDim=keptDim
            self.trainedList=[row.split(" ") for row in myList]
            self.trainedVecMat=myMat
    
    def transformToVec(self,myMat):
        '''
        transform myMat into vectors with method
        '''
        if self.method=="svd":
            if type(myMat)==sc.sparse.csr.csr_matrix:
                svdMatList=svd(myMat.A)
            else:
                svdMatList=svd(myMat)
            myMat=svdMatList[0][:,:self.keptDim]
        return myMat

    def predict(self,myArr):
        '''
        predict with CF
        '''
        if self.method in ["svd"]:
            myList=myArr.tolist()
            mySetList=[set(row.split(" ")) for row in myList]
            myMat=self.transformToMat(myArr)
            myVecMat=np.matrix(self.transformToVec(myMat))
            trainedVecMat=np.matrix(self.transformToVec(self.trainedVecMat))
            realDim=min(myVecMat.shape[1],trainedVecMat.shape[1])
            myVecMat=myVecMat[:,:realDim]
            trainedVecMat=trainedVecMat[:,:realDim]

            num=myVecMat*trainedVecMat.T
            den=np.sum((myVecMat*myVecMat.T),axis=1)*np.sum((trainedVecMat*trainedVecMat.T),axis=1).T
            den=np.sqrt(den)+0.00001*np.ones(den.shape)
            cosDisArr=np.array(num)*np.array(1/den)

            cosDisList=cosDisArr.tolist()
            maxIndexList=cosDisArr.argmax(axis=1).tolist()
            nearestTrainList=[self.trainedList[row] for row in maxIndexList]

            mixedSetList=[set(nearestTrainList[i]+myList[i].split(" ")) for i in range(len(nearestTrainList))]
            recList=[mixedSetList[rowI]-mySetList[rowI] for rowI in range(len(mySetList))]
        return recList

def itemMRR(trueY,itemPreYList):
    if trueY not in itemPreYList:
        return 0
    else:
        return 1/(itemPreYList.index(trueY)+1)

if __name__=="__main__":
    # itemVecMat,itemList=configItemBuyerMat(isTfIdf=False,itemAttr=True)

    print("loading K-Fold data ...")
    buyerItemList=np.array(pd.read_csv("data/buyerItemData.csv")["item_id"].apply(lambda x:x.split(","))).tolist()
    xArr=np.array([" ".join(row[:-1]) for row in buyerItemList])
    yArr=np.array([row[-1] for row in buyerItemList])
    kf = KFold(len(buyerItemList),n_folds=5,shuffle=False)
    mySVDCF=CF(method="svd",keptInfo=0.7)
    for i,(train_index,test_index) in enumerate(kf):

        print(i+1,"training ...")

        print("-building model and in/outputs ...")
        trainX=xArr[train_index]
        trainY=yArr[train_index]

        print("-training model ...")
        mySVDCF.train(trainX,trainY)

        print(i+1,"testing ...")
        testX=xArr[test_index]
        testY=yArr[test_index].tolist()
        preY=mySVDCF.predict(testX)

        preY=[list(preYItem) for preYItem in preY]
        MRR=np.sum([itemMRR(testY[i],preY[i]) for i in range(len(preY))])

        print("MRR:",MRR)
