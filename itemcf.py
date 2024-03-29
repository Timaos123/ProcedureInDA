import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from operator import itemgetter
import itertools
import math
from scipy import sparse
from datetime import datetime
import os
os.chdir(os.getcwd())

#############################################################################
# 将数据按ID升序顺序排下来user=0就是第一位用户                                   #
# topK为推荐物品数量                                                          #
# sim_item_num为搜索相似物品数量，比如sim_item_num=2，则只找出与商品最相似的两件商品#
#############################################################################

###################################################
# recommend_previous选择是否把已买过的物品加进推荐列表 #
###################################################
class ItemCF(object):

    def __init__(self, df, previous=True):

        print('参数初始化......')
        self.topK = 30
        self.sim_item_num = 4
        self.test_size = 0.001
        print('topK={}'.format(self.topK))
        print('sim_item_num={}'.format(self.sim_item_num))
        #print('test_size={}'.format(self.test_size))
        print('-'*30)
        self.df_user_scaler = LabelEncoder()
        self.df_item_scaler = LabelEncoder()

        self.df_user_label = np.array
        self.df_item_label = np.array
        self.train_label_user = np.array
        self.val_label_user = np.array
        self.test_label_user = np.array

        self.sub_df = pd.DataFrame()
        self.new_df = pd.DataFrame
        self.recommend_previous = previous
        print('将user-item csr矩阵转化为item相似csr矩阵......')

        # self.csrmatrix = csrmatrix

        self.csrmatrix = self.process_dataset(df)
        self.sim_item_csrmatrix = self.item_sim_matrix(self.csrmatrix)
        print('转化成功！')


    def loadfile(self, filename):

        print("读取数据中......")

        print("加载数据成功")
        pass

    def generate_dataset(self, df):

        print('划分数据为训练集跟测试集......')

        df.sort_values('buyer_admin_id',ascending=True,inplace=True)
        #############################
        # 根据机器性能适当选择前X条数据 #
        #############################
        df = df[:1000000]
        self.df_user_label = self.df_user_scaler.fit_transform(df['buyer_admin_id'])
        self.df_item_label = self.df_item_scaler.fit_transform(df['item_id'])

        self.train_label_user, self.val_label_user = train_test_split(np.unique(self.df_user_label),test_size=self.test_size)
        print('验证集用户数为:{}'.format(len(self.val_label_user)))

        element = np.array([1] * len(df))
        df_user_item_matrix = sparse.csr_matrix((element, (self.df_user_label, self.df_item_label)))

        #user-item-score矩阵
        #df_user_item_score_matrix =
        templist = list()
        for n in list(self.val_label_user):
            templist.append(np.unique(self.df_user_scaler.inverse_transform(self.df_user_label))[n])
        self.new_df = df[df.irank.isin([1])&df.buyer_admin_id.isin(templist)]

        return df_user_item_matrix

    def process_dataset(self,df):

        print('处理训练集跟测试集......')
        self.df_user_label = self.df_user_scaler.fit_transform(df['buyer_admin_id'])
        self.df_item_label = self.df_item_scaler.fit_transform(df['item_id'])
        self.test_label_user = np.unique(self.df_user_label[-len(test_data):])

        element = np.array([1] * len(df))
        df_user_item_matrix = sparse.csr_matrix((element, (self.df_user_label, self.df_item_label)))

        #user-item-score矩阵
        #df_user_item_score_matrix =

        return df_user_item_matrix



    def item_sim_matrix(self,input_csrmatrix):

        user_item_csrmatrix = input_csrmatrix
        inverseTabel = defaultdict(int)
        each_item_total_num = np.array(user_item_csrmatrix.sum(axis=0))[0, :]
        print('each_item_total_num最小值:',min(each_item_total_num))
        for user in range(user_item_csrmatrix.shape[0]):
            comb = itertools.combinations(user_item_csrmatrix.getrow(user).indices, 2)
            for index, (i, j) in enumerate(comb):
                inverseTabel.setdefault((i, j), 0)
                inverseTabel[(i, j)] += 1
                inverseTabel.setdefault((j, i), 0)
                inverseTabel[(j, i)] += 1

        X = list()
        Y = list()
        similarity = list()

        for item_pair, value in inverseTabel.items():
            X.append(item_pair[0])
            Y.append(item_pair[1])
            similarity.append(value / math.sqrt(int(each_item_total_num[X[-1]]) * int(each_item_total_num[Y[-1]])))

        for i in range(user_item_csrmatrix.shape[1]):
            X.append(i)
            Y.append(i)
            similarity.append(1)

        sim_item_csrmatrix = csr_matrix((similarity, (X, Y)), shape=(user_item_csrmatrix.shape[1], user_item_csrmatrix.shape[1]))

        return sim_item_csrmatrix

    def rank(self,user):

        sim_item = dict()
        for item in self.csrmatrix.getrow(user).indices:
            another_items = self.sim_item_csrmatrix.getrow(item).indices
            another_items_score = self.sim_item_csrmatrix.getrow(item).data
            comb = [[another_items[index], another_items_score[index]] for index in range(len(another_items))]
            comb.sort(key=itemgetter(1), reverse=True)
            sim_item[item] = comb[:self.sim_item_num]

        if self.recommend_previous:
            candidate = set(range(self.csrmatrix.shape[1]))
        else:
            user_item = list(self.csrmatrix.getrow(user).toarray()[0, :])
            candidate = set([index for index, i in enumerate(user_item) if i == 0])

        temp_recommend = dict()
        for item, another_items in sim_item.items():
            interesting = self.csrmatrix[user, item]
            for another_item, score in another_items:
                if candidate.intersection(set([another_item])):
                    temp_recommend.setdefault(another_item, 0)
                    temp_recommend[another_item] += score * interesting

        temp_recommend = [[self.df_item_scaler.inverse_transform(self.df_item_label)[list(self.df_item_label).index(key)], value] for key, value in temp_recommend.items()]
        temp_recommend.sort(key=itemgetter(1), reverse=True)

        return temp_recommend[:self.topK]

    def recommend(self,user):

        back_user = np.unique(self.df_user_scaler.inverse_transform(self.df_user_label))[user]
        final_recommend = {back_user: self.rank(user)}
        return final_recommend

    def evaluate(self):

        each_mrr = 0
        for users in set(self.val_label_user):

            for user,item_score in self.recommend(users).items():
                place = 1
                for item,score in item_score[:30]:
                    if item in self.new_df[self.new_df['buyer_admin_id'] == user].item_id.tolist():
                        each_mrr += 1/place
                    else:
                        place += 1

        mrr = each_mrr/len(self.val_label_user)
        print('MRR:{}'.format(mrr))

    def submission(self):

        for index,users in enumerate(set(list(self.test_label_user))):
            for user,item_score in self.recommend(users).items():
                if len(item_score) == 0:
                    self.sub_df[user] = np.pad(np.array([]), (0, 30), 'constant')
                elif len(item_score) < 30:
                    self.sub_df[user] = np.pad(np.array(item_score)[:, 0], (0, 30 - len(item_score)), 'edge')
                else:
                    self.sub_df[user] = np.array(item_score)[:, 0]
                print('第', index + 1, '位用户', user, '已推荐完毕')
        sub_df = self.sub_df.T
        sub_df.to_csv('submission.csv', header=None)


if __name__ == '__main__':

    #data = pd.read_csv('Antai_AE_round1_train_20190626.csv')
    data = pd.read_pickle('all_data.pkl')
    data = data[['buyer_admin_id','item_id']]
    test_data = pd.read_csv('Antai_AE_round1_test_20190626.csv')
    start = datetime.now()
    test = ItemCF(data,previous=True)
    print('-'*30)
    test.submission()
    print('总共耗时{}'.format(datetime.now() - start))
