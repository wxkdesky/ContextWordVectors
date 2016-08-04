import sys
import os
import shutil
import numpy as np
from sklearn import svm
from sklearn import metrics

def fetchData(path):
    X_train=[]
    Y_train=[]
    X_test=[]
    Y_test=[]
    files=os.listdir(path)
    root_dir=path+'/'+'features'
    try:
        os.mkdir(root_dir)
        os.mkdir(root_dir+'/train')
        os.mkdir(root_dir+'/test')
    except:
        pass
    train=True
    for file in files:
        if os.path.isdir(path+'/'+file):
            new_dir=path+'/'+file#train
            if 'train' in file:
                train=True
            else:
                train=False
            for xfile in os.listdir(new_dir):
                if os.path.isdir(new_dir+'/'+xfile):
                    new_new_dir=new_dir+'/'+xfile#certain folder
                    f_i_l_e='n'
                    for xxfile in os.listdir(new_new_dir):#certain file
                        if 'feature' in xxfile:
                            f_i_l_e=new_new_dir+'/'+xxfile
                            if train==True:
                                try:
                                    os.remove(root_dir+'/train/'+xxfile)
                                    shutil.copy(f_i_l_e,root_dir+'/train')
                                except:
                                    pass
                            else:
                                try:
                                    os.remove(root_dir+'/test/'+xxfile)
                                    shutil.copy(f_i_l_e,root_dir+'/test')
                                except:
                                    pass
                            break
    new_files_train=os.listdir(root_dir+'/train')
    count_train=0
    for xfile in new_files_train:
        count_train+=1
        for line in open(root_dir+'/train/'+xfile):
            if len(line)>10:
                line1=str(line)[1:len(line)-2]
                X_train.append(line1)
                Y_train.append(count_train)
    print('training files loaded!')
    new_files_test=os.listdir(root_dir+'/test')
    count_test=0
    for xfile in new_files_test:
        count_test+=1
        for line in open(root_dir+'/test/'+xfile):
            if len(line)>10:
                line1=str(line)[1:len(line)-2]
                X_test.append(line1)
                Y_test.append(count_test)
    print('testing files loaded!')
    print('svm algorithm starts......')
    adjustArray(X_train)
    adjustArray(X_test)
    clf=svm.SVC(kernel='linear')
    model=clf.fit(X_train,Y_train)
    predicted=model.predict(X_test)
    print(metrics.classification_report(Y_test, predicted))
    print(metrics.confusion_matrix(Y_test, predicted))
    print('all tasks done!')


def adjustArray(X_train):
    for i in range(len(X_train)):
        item=[]
        for it in X_train[i].split(','):
            item.append(float(it))
        X_train[i]=item

if __name__=='__main__':
    print('hello world')
    path='./parameter'
    fetchData(path)