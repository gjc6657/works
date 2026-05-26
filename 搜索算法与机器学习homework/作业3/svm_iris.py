import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
先试着用所有特征分类
"""

iris = load_iris()
X = iris.data
print(X.shape,X)
y = iris.target
print(y.shape,y)

"""
数据划分
"""

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = StandardScaler()#数据标准化

X_train = scaler.fit_transform(X_train)#训练集计算均值和标准差
X_test = scaler.transform(X_test)#测试集用先前的标准化规则

svm_inside = svm.SVC(kernel='rbf',C=1,gamma='auto')
#SVC:支持向量分类器；kernel：核函数，用于映射高维度，rbf为径向基函数，用的比较多；
#C=1为惩罚参数，越小泛化越强；gamma控制变量影响的范围，即支持向量的影响，越大边界越复杂
#取gamma合适的量，不能过拟合也不能欠拟合
#'auto'关键字是1/nfeature，这里是1/4
svm_inside.fit(X_train, y_train)
print("测试得分：",svm_inside.score(X_test, y_test))
print("预测：",svm_inside.predict([[7,5,2,0.5],[7,5,4,2]]))

"""
四维的我还没想好怎么用plt输出
"""

"""
再试一下用两个特征分类，
"""

X2 = X[:,:2]
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,y,test_size=0.2,random_state=42)#划分训练集和测试集
scaler2 = StandardScaler()

X2_train = scaler2.fit_transform(X2_train)
X2_test = scaler2.transform(X2_test)

svm_inside.fit(X2_train, y2_train)
print("二维测试得分：",svm_inside.score(X2_test, y2_test))
print("预测：",svm_inside.predict([[7,5],[7,4]]))

"""
尝试一下plt输出
"""

x_min, x_max = X2_train[:, 0].min() - 1, X2_train[:, 0].max() + 1
y_min, y_max = X2_train[:, 1].min() - 1, X2_train[:, 1].max() + 1

h = 0.02

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

Z = svm_inside.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1,1,1)
plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.8)
plt.scatter(X2[:, 0],X2[:, 1],c=y,cmap=plt.cm.Paired,edgecolors='k')
plt.scatter(svm_inside.support_vectors_[:, 0],svm_inside.support_vectors_[:, 1],s=200,facecolors='none',edgecolors='red')
plt.xlabel('Sepal width')
plt.ylabel('Petal length')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('SVC with RBF kernel')

plt.show()

"""
这段plt输出是跟着gpt写的，plt图像输出我觉得我还得回去再学一学
"""