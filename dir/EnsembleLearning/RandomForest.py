import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
import pandas as pd

if __name__=='__main__':
    data=np.array(pd.read_csv('D:/iris.csv'))
    xtrain,xtest,ytrain,ytest=train_test_split(data[1:,:-1],data[1:,-1],test_size=0.2)
    classifier=RandomForestClassifier(n_estimators=10)
    kf=KFold(n_splits=5,shuffle=True,random_state=0)
    trainAccuracy=0#训练集的正确率
    valueAccuracy=0#验证集的正确率
    for (train_index,test_index) in kf.split(xtrain,ytrain):#返回训练集和测试集的索引
        trainData=xtrain[train_index]
        trainValue=ytrain[train_index]
        testData=xtrain[test_index]
        testValue=ytrain[test_index]
        classifier.fit(trainData,trainValue)
        trainAccuracy+=classifier.score(trainData,trainValue)
        valueAccuracy+=classifier.score(testData,testValue)
    print("训练集的正确率：{}".format(trainAccuracy/5))
    print("验证集的正确率：{}".format(valueAccuracy/5))
      
    print("测试集的准确率：{}".format(classifier.score(xtest,ytest)))
