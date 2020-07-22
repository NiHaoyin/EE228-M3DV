# EE228-M3DV
本项目为EE228 倪冰冰老师机器学习的课程大作业 3D医学影像分类  
数据集为病人某一器官的3D影像，分类标签为是否含有某种基因
# 文件说明 
DataProcess.py:包含数据集读取和数据增强。直接调用load_data()即可。load_data(augment = False) 读取原始数据集和标签；load_data(augment = True)读取数据增强后的数据集合标签  
main.py：进行训练。  
test.py: 生成预测结果。  
0617.csv:我的预测结果，最终分数为0.77  
Model/Densenet:我采用的神经网络模型  

 
