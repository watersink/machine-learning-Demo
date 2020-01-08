from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.datasets import load_wine
from sklearn.externals import joblib



wine = load_wine()
Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data,wine.target,test_size=0.3)

forest = RandomForestClassifier(random_state=0,n_estimators=25)
forest.fit(Xtrain,Ytrain)
score_r = forest.score(Xtest,Ytest)

print("Random Forest:{}".format(score_r))
# 保存模型
joblib.dump(forest, "forest.model") 
# 加载模型
#forest = joblib.load("forest.model")



forest_score = cross_val_score(forest,wine.data,wine.target,cv=10).mean()
print("10 folder val score: ",forest_score)
