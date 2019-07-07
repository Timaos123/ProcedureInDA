# ProcedureInDA 使用说明

大致流程分布包括：A（数据描述及预处理）、B（模型训练）、C（模型预测）[、D（模型整合）]
目前所拥有的文件包括：

data：存储数据（不放到github上）

log：深度学习日志数据

model：模型数据（训练后存储）

A0_originalDataDes.py：原始数据描述

A1_preprocessingForML.py：传统机器学习预处理（基于矩阵的数据结构）

A2_preprocessingForDL.py：深度学习预处理（基于张量的数据结构）

B0_trainML.py：训练传统机器学习

B1_trainDL.py：训练深度学习

C0_predictWithML.py：传统机器学习预测

C1_predictWithDL.py：深度学习预测	

可在每一栏后新添自己相关新文件，并附带序号（eg.在B1后新添B2_trainSelf.py）
