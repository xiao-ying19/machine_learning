import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split,KFold
import pandas as pd


if __name__=='__main__':
    data=np.array(pd.read_csv('iris.csv',encoding='utf-8'))
    x_train,x_test,y_train,y_test=train_test_split(data[:,:-1],data[:,-1],test_size=0.2)
    classifier=svm.SVC(kernel='rbf',gamma=0.1,decision_function_shape='ovo',C=0.1)
    #classifier.fit(x_train,y_train)
    kf=KFold(n_splits=5,shuffle=True,random_state=0)
    trainAccuracy=0#训练集的正确率
    valueAccuracy=0#验证集的正确率
    for (train_index,test_index) in kf.split(x_train,y_train):#返回训练集和测试集的索引
        trainData=x_train[train_index]
        trainValue=y_train[train_index]
        testData=x_train[test_index]
        testValue=y_train[test_index]
        classifier.fit(trainData,trainValue)
        trainAccuracy+=classifier.score(trainData,trainValue)
        valueAccuracy+=classifier.score(testData,testValue)
    print("训练集的正确率：{}".format(trainAccuracy/5))
    print("验证集的正确率：{}".format(valueAccuracy/5))
      
    print("测试集的准确率：{}".format(classifier.score(x_test,y_test)))

    
    
    

    
    

