import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
def data_operate():
    '''
    对文件里的数据集进行处理,划分训练集和测试集
    '''
    label_enoder=LabelEncoder()
    onehot_encoder=OneHotEncoder()
    np.set_printoptions(suppress=True)#不用科学计数法表示
    flowers=np.array(pd.read_csv("iris.csv"))
    np.random.shuffle(flowers) #打乱顺序
    training_data=flowers[:100,1:5] #训练集的输入

    training_label=flowers[:100,5]  #训练集的分类
    training_label=label_enoder.fit_transform(training_label)
    training_label=np.reshape(training_label,(-1,1))
    training_label=onehot_encoder.fit_transform(training_label).toarray()

    test_data=flowers[100:,1:5]  #测试集的输入
    test_label=flowers[100:,5]  #测试集的分类
    test_label=label_enoder.fit_transform(test_label)
    test_label=np.reshape(test_label,(-1,1))
    test_label=onehot_encoder.fit_transform(test_label).toarray()
    return training_data,training_label,test_data,test_label
    

def sigmoid(n):
    return 1/(1+np.exp(0-n))


def BP_NetWorking(training_data,training_label,learn_rate,error):
    '''
    BP神经网络算法
    training_data：训练数据
    training_label：类别
    learn_rate:学习率
    error:误差
    '''
    #初始化
    d=len(training_data[0])#输入神经元个数
    q=2*d+1                #隐层神经元个数
    l=len(np.array(list(set([tuple(t)for t in training_label]))))#输出层神经元个数
    w1=np.random.rand(d,q) #输入层到隐层的权值，4*9,
    w2=np.random.rand(q,l) #隐层到输出层的权值，9*3
    b1=np.random.rand(1,q)#隐层元素的阈值,1*9
    b2=np.random.rand(1,l)#输出层元素的阈值，1*3
    #print(type(training_data))
    epoch=0
    while(True):
        Error=[]#累积误差
        for i in range (len(training_data)):
            #计算当前样本的输出
            x1=np.dot(training_data[i:i+1,:],w1)#隐层神经元的输入，1*9
            x2=sigmoid(x1-b1) #隐层神经元的输出,1*9
            y1=np.dot(x2,w2)#输出层的输入,1*3
            y2=sigmoid(y1-b2)#当前样本的输出,1*3
            #计算均方误差K
            Err=np.dot((y2-training_label[i:i+1,:]),(y2-training_label[i:i+1,:]).T)/2
            Error.append(Err)
            #计算输出层神经元的梯度项g
            g=y2*(np.ones((1,3))-y2)*(training_label[i:i+1,:]-y2)
            #计算隐层神经元的梯度项e
            a=np.dot(w2,g.T)#9*1
            e=x2*(np.ones((1,9))-x2)*a.T  #1*9
            #print(np.shape(e))
            #更新连接权与阈值
            w2=w2+learn_rate*np.dot(x2.T,g)#更新隐层到输出层的权值
            b2=b2-learn_rate*g #更新输出层的阈值
            w1=w1+learn_rate*np.dot(training_data[i:i+1,:].T,e)#更新输入层到隐层的权值
            b1=b1-learn_rate*e#更新隐层神经元的阈值
            epoch+=1
        if(sum(Error)/len(training_data)<error or epoch>50000):
            break
    return w2,b2,w1,b1#返回连接权和阈值

def test(test_data,test_label,w1,w2,b1,b2):
    '''
    测试
    '''
    x1=np.dot(test_data,w1)#隐层神经元的输入，50*9
    x2=sigmoid(x1-b1) #隐层神经元的输出,50*9
    y1=np.dot(x2,w2)#输出层的输入,50*3
    y2=sigmoid(y1-b2)#当前样本的输出,50*3

    n_rows=test_label.shape[0]#行
    n_cols=test_label.shape[1]#列

    OutPut=np.empty(shape=(n_rows,n_cols),dtype=int)

    for i in range (n_rows):
        for j in range(n_cols):
            if(y2[i][j]>0.5):
                OutPut[i][j]=1
            else:
                OutPut[i][j]=0
    # print(OutPut)
    # print(test_label)
    count=0
    for i in range(len(OutPut)):
        if(OutPut[i]==test_label[i]).all():
            count+=1
    return count/n_rows

if __name__=='__main__':
    training_data,training_label,test_data,test_label=data_operate()
    w2,b2,w1,b1=BP_NetWorking(training_data,training_label,0.2,0.001)
    #测试
    corr_rate=test(test_data,test_label,w1,w2,b1,b2)
    print("正确率：{:.2f}%".format(corr_rate*100))

