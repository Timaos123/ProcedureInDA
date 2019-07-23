#1.先将时间划分为month，day，hour，再将hour划分为早上，中午，下午，晚上，深夜。（总计8个特征）
#2.再算出每个商店的平均消费水平store_average_price,商品数量store_item_num,以及商店人气(按购买次数来算)store_buyer_num。（总计3个特征）
#3.然后算出每位buyer的平均消费水平buyer_average_price,购物次数buyer_shopping_num（总计2个特征）
#4.再按消费水平(price范围是1到2w嘛)初步考虑将buyer从高到低分为A、B、C、D、E五个消费档次buyer_shopping_level（总计1个特征）
#5.最后设计兴趣指标函数f(x),算出user对item的最终得分,比如storeid=2333的buyer_average_price处于buyer_admin_id=3222的消费水平内（假如为C）
，则f(x)的值加1，如处于以外，则为0，即保持不变

"""
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
