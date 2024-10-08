# author: baiCai
# 导包
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# 加载数据
data = load_iris()
# print(data) # 返回的字典
# 划分数据集 8：2
x_train,x_test,y_train,y_test = model_selection.train_test_split(data['data'],data['target'],test_size=0.2,random_state=22)
# print(x_train.shape) # 120,4
# print(y_train.shape) # 120,

# # 创建模型
# model = KNeighborsClassifier(n_neighbors=5)
# model.fit(x_train,y_train)
# # 评估
# score = model.score(x_test,y_test)
# print('测试集准确率：',score)
# # 评估2
# y_predict = model.predict(x_test)
# print('测试集对比真实值和预测值：',y_predict == y_test)

# 探究k值影响
model_new = {
    KNeighborsClassifier(n_neighbors=2),
    KNeighborsClassifier(n_neighbors=3),
    KNeighborsClassifier(n_neighbors=4),
    KNeighborsClassifier(n_neighbors=5),
    KNeighborsClassifier(n_neighbors=6),
    KNeighborsClassifier(n_neighbors=7),
    KNeighborsClassifier(n_neighbors=8),
    KNeighborsClassifier(n_neighbors=9),
    KNeighborsClassifier(n_neighbors=10),
}
score_list = []
for model in model_new:
    model.fit(x_train,y_train)
    score = model.score(x_test,y_test)
    score_list.append(score)
# 画出图形
# 处理中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure()
plt.bar(range(2,11),score_list)
plt.title('不同K值准确率')
plt.show()
# print(score_list)

# 画出训练集
# 处理中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.figure()
# c = ['r','b','g']   # 设置三个颜色
# color = [c[y] for y in y_train] # 为不同的标签设置颜色，比如0--r--红色
# plt.scatter(x_train[:,0],x_train[:,1],c=color)
# plt.title('训练集图')
# plt.xlabel('花萼长')
# plt.ylabel('花萼宽')
# plt.show()
