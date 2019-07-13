#测试用，后面会删，或改造
from surprise import Dataset
from surprise import KNNBasic
from surprise import Reader


reader = Reader()

#数据集是movielens，其中rating是点击次数

data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
all_trainset = data.build_full_trainset()

algo = KNNBasic(k=40,min_k=3,sim_options={'user_based': True}) # sim_options={'name': 'cosine','user_based': True} cosine/msd/pearson/pearson_baseline
algo.fit(all_trainset)

#输出前topk个相似用户，uid是指定用户id
def getSimilarUsers(top_k,u_id):
    user_inner_id = algo.trainset.to_inner_uid(u_id)
    user_neighbors = algo.get_neighbors(user_inner_id, k=top_k)
    user_neighbors = (algo.trainset.to_raw_uid(inner_id) for inner_id in user_neighbors)
    return user_neighbors

list(getSimilarUsers(5,1))


# 官方数据集下载地址http://grouplens.org/datasets/movielens/

# 数据集简介如下：

# MovieLens 100K Dataset
# Stable benchmark dataset. 100,000 ratings from 1000 users on 1700 movies. Released 4/1998.


# MovieLens 1M Dataset
# Stable benchmark dataset. 1 million ratings from 6000 users on 4000 movies. Released 2/2003.


# MovieLens 10M Dataset
# Stable benchmark dataset. 10 million ratings and 100,000 tag applications applied to 10,000 movies by 72,000 users. Released 1/2009.


# MovieLens 20M Dataset
# Stable benchmark dataset. 20 million ratings and 465,000 tag applications applied to 27,000 movies by 138,000 users. Released 4/2015.


# MovieLens Latest Datasets
# Small: 100,000 ratings and 6,100 tag applications applied to 10,000 movies by 700 users. Last updated 1/2016.
# Full: 22,000,000 ratings and 580,000 tag applications applied to 33,000 movies by 240,000 users. Last updated 1/2016.


# MovieLens Tag Genome Dataset
# 11 million computed tag-movie relevance scores from a pool of 1,100 tags applied to 10,000 movies.
