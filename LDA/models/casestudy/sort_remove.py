import sys
from nltk.stem.lancaster import LancasterStemmer as ls
from nltk.stem.porter import PorterStemmer as ps
import enchant

def Modify(suffix='./wikicorp.201004.m2.txt',stopWords=['i','a','about','an','as','are','at','be','by','com','de','en','for','from','how','in','is','it','la','of','on','or','that','the','this','to','was','what','when','where','who','will','with','www']):
    newFileName='wikicorp.201004.m3.txt'
    newFile=open(newFileName,'w')
    n=0;
    #st=ls()
    st2=ps()
    d=enchant.Dict("en_US")
    for line in open(suffix):
        n+=1
        x=line.split()
        for item in range(len(x)):
            if x[item] in stopWords:
                x[item]=''
                continue
            if d.check(x[item]):
                x[item]=st2.stem(x[item])
            else:
                x[item]=''
            #x[item]=st.stem(x[item])
        x=sorted(x)
        for xline in range(len(x)-1):
            if x[xline]!='':
                newFile.write(x[xline]+' ')
        if x[-1]!='':            
            newFile.write(x[-1])
        else:
            newFile.seek(-1,2)
        if n!=3035070:
            newFile.write('\n')
    newFile.close()

if __name__=='__main__':
    #nltk.download()
    Modify()
