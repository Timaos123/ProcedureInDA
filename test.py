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
