from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_wine,load_boston
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
 

#分类
wine = load_wine()
#必须做标准化
scaler = StandardScaler() # 标准化转换
scaler.fit(wine.data)  # 训练标准化对象
wine.data= scaler.transform(wine.data)  
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target,test_size=0.3)

mlp = MLPClassifier(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(16,64,128), random_state=1,max_iter=2000,verbose=10,learning_rate_init=0.001)
mlp.fit(Xtrain, Ytrain)
score_r =mlp.score(Xtest,Ytest)

print ("test score: ",score_r)
print ("mlp.n_layers_: ",mlp.n_layers_)
print ("mlp.n_iter_: ",mlp.n_iter_)
print ("mlp.loss_: ",mlp.loss_)
print ("mlp.out_activation_: ",mlp.out_activation_)


# 保存模型
joblib.dump(mlp, "mlp.model") 
# 加载模型
mlp = joblib.load("mlp.model")



# 对测试集做预测
y_pred = mlp.predict(Xtest)
#print(Ytest,"    ",y_pred)
predictions = [round(value) for value in y_pred]
 
# 评估预测结果
accuracy = accuracy_score(Ytest, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))





#回归 
boston = load_boston()  # 加载波士顿房价回归数据集

#必须做标准化
scaler = StandardScaler() # 标准化转换
scaler.fit(boston.data)  # 训练标准化对象
boston.data= scaler.transform(boston.data) 
X1, y1 = shuffle(boston.data, boston.target, random_state=13) # 将数据集随机打乱
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.1, random_state=0)  # 划分训练集和测试集.test_size为测试集所占的比例


mlp = MLPRegressor(solver='sgd', activation='relu',alpha=1e-4,hidden_layer_sizes=(16,64,128), random_state=1,max_iter=2000,verbose=10,learning_rate_init=0.001)
mlp.fit(X_train, y_train)
score_r =mlp.score(X_test,y_test)

print ("test score: ",score_r)
print ("mlp.n_layers_: ",mlp.n_layers_)
print ("mlp.n_iter_: ",mlp.n_iter_)
print ("mlp.loss_: ",mlp.loss_)
print ("mlp.out_activation_: ",mlp.out_activation_)


# 保存模型
joblib.dump(mlp, "mlp.model") 
# 加载模型
mlp = joblib.load("mlp.model")



# 对测试集做预测
y_pred = mlp.predict(X_test)
#print(y_test, y_pred)
# 评估预测结果
print('mlp regress mse: ',mean_squared_error(y_test, y_pred))

