import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import f1_score
import pandas as pd


if __name__=='__main__':
    data=np.array(pd.read_csv('D:/iris.csv'))
    xtrain,xtest,ytrain,ytest=train_test_split(data[1:,:-1],data[1:,-1],test_size=0.2)
    model=lgb.LGBMClassifier()
    param_grid={
        'learning_rate':[0.1,0.5,0.8],
        'n_estimators':[30,40,50]
    }
    model=GridSearchCV(model,param_grid)
    model.fit(xtrain,ytrain)
    ypre=model.predict(xtest)
    print(f1_score(ytest,ypre,average='weighted'))