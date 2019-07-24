"""
1.先将时间划分为month，day，hour，再按hour划分时间段，早上、中午、下午、晚上。以及按buyer来划分出经常购买的时段（众数）。（总计8个特征）
2.再算出每个商店的平均消费水平store_average_price,商品数量store_item_num,以及商店人气(按购买次数来算)store_buyer_num。（总计3个特征）
3.然后算出每位buyer的平均消费水平buyer_average_price,购物次数buyer_shopping_num（总计2个特征）
4.再按消费水平(price范围是1到2w嘛)初步考虑将buyer从高到低分为A、B、C、D、E五个消费档次buyer_shopping_level（总计1个特征）
5.最后设计兴趣指标函数f(x),算出user对item的最终得分,比如storeid=2333的buyer_average_price处于buyer_admin_id=3222的消费水平内（假如为C）
,则f(x)的值加1，如处于以外，则为0，即保持不变

该user-item矩阵为未建立兴趣指标函数前的矩阵，例如（B,b）=2意为B用户买了b商品2次。
	  		Item	a 	b 	c 	d 	e
		User	A 	1 	1 	0 	1 	0
			B 	0 	2 	1 	0 	1
			C 	0 	0 	1 	1 	0	
			D 	0 	1 	1 	1 	0
			E 	1 	0 	0 	3 	0
假设建立好兴趣指标函数f(x)后，user-item矩阵可能变为如下矩阵,即从"user-item-购买次数"矩阵变为"user-item-score"矩阵
之后算法照旧运行（test.py）➡得出结果➡提交
	 		 Item	a 	b 	c 	d 	e
		User	A 	3 	4 	0 	3 	0
			B 	0 	5 	2 	0 	1
			C 	0 	0 	3 	6 	0	
			D 	0 	2 	4 	2 	0
			E 	1 	0 	0 	6 	0

"""
import pandas as pd
import pickle
from datetime import datetime
from scipy import stats
test=pd.read_csv('drive/tianchi/Antai_AE_round1_test_20190626.csv')
item=pd.read_csv('drive/tianchi/Antai_AE_round1_item_attr_20190626.csv')
train=pd.read_csv('drive/tianchi/Antai_AE_round1_train_20190626.csv')

start = datetime.now()

df=pd.concat([train,test],ignore_index=True)
all_data=pd.merge(df,item,on=['item_id'],how='left')
del df
all_data['create_order_time'] = all_data.create_order_time.apply(lambda x:pd.to_datetime(x))
all_data['hour']=all_data['create_order_time'].dt.hour
all_data['date']=all_data['create_order_time'].dt.day
all_data['month']=all_data['create_order_time'].dt.month

all_data.drop(['create_order_time'],axis=1,inplace=True)
all_data['cate_id'].fillna(0,inplace=True)
all_data['store_id'].fillna(0,inplace=True)
all_data['item_price'].fillna(0,inplace=True)

all_data['buyer_country_id']=all_data['buyer_country_id'].astype('category')
all_data['buyer_admin_id']=all_data['buyer_admin_id'].astype('int32')
all_data['item_id']=all_data['item_id'].astype('int32')
all_data['irank']=all_data['irank'].astype('int8')
all_data['cate_id']=all_data['cate_id'].astype('int32')
all_data['store_id']=all_data['store_id'].astype('int32')
all_data['hour']=all_data['hour'].astype('int8')
all_data['month']=all_data['month'].astype('int8')
all_data['date']=all_data['date'].astype('int8')

#商店
group = all_data.groupby('store_id')['item_price'].agg({'store_avg_price':'mean'})
group.reset_index(inplace=True)
all_data = pd.merge(all_data,group,on='store_id',how='left')

group = all_data.groupby('store_id')['item_id'].agg({'store_buyer_num':'count'})
group.reset_index(inplace=True)
all_data = pd.merge(all_data,group,on='store_id',how='left')

group = all_data.groupby('store_id')['item_id'].agg({'store_item_num':lambda x :len(x.unique())})
group.reset_index(inplace=True)
all_data = pd.merge(all_data,group,on='store_id',how='left')

#买家
group = all_data.groupby('buyer_admin_id')['item_price'].agg({'buyer_average_price':'mean'})
group.reset_index(inplace=True)
all_data = pd.merge(all_data,group,on='buyer_admin_id',how='left')

group = all_data.groupby('buyer_admin_id')['item_id'].agg({'buyer_shopping_num':'count'})
group.reset_index(inplace=True)
all_data = pd.merge(all_data,group,on='buyer_admin_id',how='left')

all_data.to_pickle('drive/tianchi/all_data.pkl')
print('总共耗时{}'.format(datetime.now() - start))

#all_data = pd.read_pickle('drive/tianchi/all_data.pkl')
#all_data.info(null_counts=True)
