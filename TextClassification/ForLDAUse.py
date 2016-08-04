import sys
import os
import shutil

def formatLDA(path):
    newpath=path+'/final'
    os.mkdir(newpath)
    files=os.listdir(path)
    for file in files:
        if not os.path.isdir(path+'/'+file):
            newfile=open(newpath+'/'+file+'.final','w')
            total=0
            count=0
            for line in open(path+'/'+file):
                total+=1
            newfile.write(str(total-1)+'\n')
            for line in open(path+'/'+file):
                count+=1
                if count<total-1:
                    newfile.write(line)
                elif count==total-1:
                    line=str(line).replace('\n','')
                    newfile.write(line)
            newfile.close()
            print('in '+path+':'+file+' is transformed')
    print(path+' is finished')


if __name__=='__main__':
    path1='./resultfile/train'
    path2='./resultfile/test'
    formatLDA(path1)
    formatLDA(path2)