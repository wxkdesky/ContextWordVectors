import sys
from scipy import stats
import time

def extractPwt(suffix='./model-final.phi'):
    Pwt_collection=[]
    for line in open(suffix):           
        item=[]
        for i,p in enumerate(line.split()):
        #item.append((i,float(p)))
            item.append(float(p))
        Pwt_collection.append(item)
    print('Pwt finished!length is '+str(len(Pwt_collection[0])))
    return Pwt_collection
def spearman():
        x=[]
        y=[]
        for line in open('./simple_test3.test'):
            l=line.split()
            x.append(float(l[0])*1e12)
            y.append(float(l[1]))
        rho,pval= stats.spearmanr(x,y)
        otherinfo=' no noise'
        file=open('./final_result_spearmanr.txt','a')
        file.write('pho is '+str(rho)+' pval is '+str(pval)+' finish time '+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))+' top '+str(2)+ otherinfo+'\n')
        file.close()

if __name__=='__main__':
    #extractPwt()
    spearman()