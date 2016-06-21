#By Wishcome
import sys
import numpy as np
import enchant
import string
import re
import time
from nltk.stem.lancaster import LancasterStemmer as ls
from nltk.stem.porter import PorterStemmer as ps
from scipy import stats

class preprocessing:
    P=dict()
    word_pair=[]
    collection=[]#2003 pair text
    Pwt_collection=[]
    Ptd_collection=[]
    wordmap_collection=[]
    topicvector_collection=[]
    mag_topic_collection=[]
    index_collection=[]#pairs satisfied
    avg_sim_total=[]
    wordvector_dict=dict()

    top=1#z1,z2..

    def __init__(self):
        file=open('./final_avg.txt','w')
        file.close()
    #i represents the line sequence of the word begins with 0 in wordmap.txt(the first line is ignored),index of the collection is the id of the topic
    def extractPwt(self,suffix='./model-01800.phi'):
        for line in open(suffix):           
            item=[]
            for i,p in enumerate(line.split()):
                #item.append((i,float(p)))
                item.append(float(p))
            self.Pwt_collection.append(item)
        print('Pwt finished!length is '+str(len(self.Pwt_collection[0])))
        return self.Pwt_collection

    #i represents the id of the topic,index of collection is the index of doc
    def extractPtd(self,suffix='./model-01800.theta'):
        for line in open(suffix):
            item=[]
            for i,p in enumerate(line.split()):
                #item.append((i,float(p)))
                item.append(float(p))
            self.Ptd_collection.append(item)
        print('Ptd finished!length is '+str(len(self.Ptd_collection[0])))
        return self.Ptd_collection

    #index of readWordMap collection is the i of extractPwt collection
    def readWordMap(self,suffix='./wordmap.txt'):
        count=0;
        for line in open(suffix):
            count+=1
            if count>1:
                self.wordmap_collection.append(line.split()[0])# the true id of the word in the wordmap.txt is discarded cause it can be replaced with index of readWordMap collection
        #$self.wordmap_collection=list(set(self.wordmap_collection))
        print('totoal words in map '+str(len(self.wordmap_collection)))
        return self.wordmap_collection

    def readNewWordMap(self,suffix='new_wordmap.txt'):
        self.wordmap_collection=[]
        self.readWordMap(suffix)


    def readWordvector(self,suffix='./word_vector.txt'):
        for line in open(suffix):
            name=''
            item=[]
            for i in line.split():
                if i=='nan':
                    name=i
                    continue
                try:
                    v=float(i)
                    item.append(v)
                except:
                    name=i
            self.wordvector_dict[name]=item
        print('word vector count '+str(len(self.wordvector_dict)))
        return self.wordvector_dict

    def readTopicvector(self,suffix='./topic_vector.txt'):
        for line in open(suffix):
            item=[]
            for i in line.split():
                try:
                    v=float(i)
                    item.append(v)
                except:
                    print('Error in converting str to float in %s' %suffix)
            self.topicvector_collection.append(item)
            #print(len(item))
        return self.topicvector_collection

    #compute topic vector
    def magCalc(self,topic):
        for item in topic:
            self.mag_topic_collection.append(np.linalg.norm(item))
            #self.mag_topic_collection.append(1)
        return self.mag_topic_collection

    #test function
    def writeColumn(topic,n):
        file=open('topic.txt','w')
        for i in range(len(topic)):
            if i==n:
                for item2 in topic[n]:
                    file.write(str(item2)+'\n')
        file.close()

    #topic: topic vector collection
    def normalize_topic(self,topic):
        new_topic=[]
        for t in topic:
            topic_1=np.abs(t)
            max_num=np.max(topic_1)
            topic_2=[float(x)/max_num for x in t]
            #print(np.max(topic_2))
            new_topic.append(topic_2)
        return new_topic
        #return topic

    #word: word vector dictionary
    def normalize_word(self,word):
        new_word=dict()
        for key in word:
            word_1=np.abs(word[key])
            max_num=np.max(word_1)
            new_word[key]=[float(x)/max_num for x in word[key]]
        return new_word
        #return word

    '''
    mtopic:magnitude of topic
    nword:normalized word vector collection
    ntopic:normalized topic vector collection
    word_name:word itself
    topic_index: index of topic vector collection

    '''
    def buildFinalVector(self,mtopic,nword,ntopic,word_name,topic_index,xtopicvector):
        word_vec=nword[word_name]
        if(len(xtopicvector)<=2):
            topic_vec=ntopic[topic_index]
            mag=mtopic[topic_index]
            final_word_vec=[float(mag)*x for x in word_vec]
            for item in topic_vec:
                final_word_vec.append(item)
        else:
            mag=np.linalg.norm(xtopicvector)
            final_word_vec=[float(mag)*x for x in word_vec]
            for item in xtopicvector:
                final_word_vec.append(item)
        return final_word_vec

    '''

    c_index:the index of docs collection
    word_name:word itself
    topic_range:topic numbers in LDA

    '''
    def pzIwc(self,c_index,word_name,topic_range=200):
        self.P[word_name+'_'+str(c_index)]=[]
        word_index=0
        topic_collection=[]
        top_topic_collection=[]
        tmp_index=[]
        topic_index_array=[]
        sum=float(0)
        for item in range(len(self.wordmap_collection)):
            if self.wordmap_collection[item]==word_name:
                word_index=item
                break
        for iter in range(topic_range):
            tmp_index.append(iter)
            try:
                topic_collection.append((iter,np.abs(self.Pwt_collection[iter][word_index])))                
            except:
                print('Can not find word_index in Pwt!Set it to 0!')
                topic_collection.append((iter,0))
        topic_collection=sorted(topic_collection,cmp=None,key=lambda x_tuple:x_tuple[1],reverse=True)
        for iter in range(self.top):
            top_topic_collection.append((topic_collection[iter][0],topic_collection[iter][1]))
            topic_index_array.append(topic_collection[iter][0])
        for iter in range(2,len(topic_collection)):
            sum=sum+topic_collection[iter][1]
        #top_topic_collection[2]=(-1,sum/(len(topic_collection)-2))
        for iter in top_topic_collection:
            tmp_index.remove(iter[0])
            topic_index=iter[0]
            Pwt=self.Pwt_collection[topic_index][word_index]
            Ptd=self.Ptd_collection[c_index][topic_index]
            self.P[word_name+'_'+str(c_index)].append(Pwt*Ptd)#+'_'+str(topic_index)+'_'+str(c_index)
        self.P[word_name+'_'+str(c_index)].append(sum/(len(topic_collection)-self.top))#+'_'+str(-1)+'_'+str(c_index)
        avg3_tv=np.zeros_like(self.topicvector_collection[0])
        for item in tmp_index:
            avg3_tv=np.add(self.topicvector_collection[item],avg3_tv)
        avg3_tv=avg3_tv/len(tmp_index)
        return (topic_index_array,avg3_tv.tolist())

    def resolve(self,path='./ratings.txt'):
        file=open('./ratings1.txt','w')
        for line in open(path):
            line=str(line).translate(None,'\n')
            xline=line.split('\t')
            t=xline[8:18]
            t=[float(x) for x in t]
            max=np.max(t)
            min=np.min(t)
            xline.append(str(max))
            xline.append(str(min))
            self.collection.append([x.lower() for x in xline])#id1,word1,POS1,word2,POS2,word1Context,word2Context,AvgRating,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10
            file.write(xline[5]+'\n')
            file.write(xline[6]+'\n')
        file.close()
        return self.collection


    def Modify(suffix='./ratings1.txt',stopWords=['i','a','about','an','as','are','at','be','by','com','de','en','for','from','how','in','is','it','la','of','on','or','that','the','this','to','was','what','when','where','who','will','with','www']):
        newFileName='ratings2.txt'
        newFile=open(newFileName,'w')
        n=0;
        #st=ls()
        #pat=re.compile('[a-zA-Z\s]')
        pat='[^a-zA-Z\s]'
        st2=ps()
        d=enchant.Dict("en_US")
        for line in open(suffix):
            n+=1
            x=line.split()
            for item in range(len(x)):
                if x[item] in stopWords:
                    x[item]=''
                    continue
                if d.check(x[item]) and x[item] not in string.punctuation and re.match(pat,x[item])==None:
                    x[item]=st2.stem(x[item].lower())
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
            if n!=4006:
                newFile.write('\n')
        newFile.close()

    def build_test_pair(self):        
        for item in self.collection:
            self.word_pair.append((item[1],item[3]))
        return self.word_pair

    def find_valid_pair(self,path='./ratings2.txt',final_path='./ratings_final.txt'):
        st=ps()
        cp_collection=[]
        docs=[]
        new_docs=[]
        new_word_pair=[]
        key=self.wordvector_dict.keys()
        for line in open(path):
            docs.append(line)
        for index in range(len(self.word_pair)):
            self.word_pair[index]=(st.stem(self.word_pair[index][0]),st.stem(self.word_pair[index][1]))
            if self.word_pair[index][0] in docs[index*2] and self.word_pair[index][1] in docs[index*2+1] and self.word_pair[index][0] in self.wordmap_collection and self.word_pair[index][1] in self.wordmap_collection and self.word_pair[index][0] in key and self.word_pair[index][1] in key:
                self.index_collection.append(index)
                cp_collection.append(self.collection[index])
        #return index_collection
        for i in self.index_collection:
            new_docs.append(docs[i*2])
            new_docs.append(docs[i*2+1])
            new_word_pair.append(self.word_pair[i])
        self.word_pair=new_word_pair
        file=open(final_path,'w')
        file.write(str(len(new_docs))+'\n')
        for item in range(len(new_docs)-1):
            file.write(new_docs[item])
        pat=re.compile('\n')
        new_docs[-1]=re.sub(pat,'',new_docs[-1])
        file.write(new_docs[-1])
        file.close()
        self.collection=cp_collection
        #return new_docs

    def similarity(self,vec1,vec2):
        return np.dot(vec1,vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))

    def avg_calc(self,mtopic,nword,ntopic):
        st=ps()
        for item in range(len(self.collection)):
            word1=st.stem(self.collection[item][1])
            word2=st.stem(self.collection[item][3])
            word1_3=self.pzIwc(2*item,word1,topic_range=200)
            word2_3=self.pzIwc(2*item+1,word2,topic_range=200)
            W1=[]
            W2=[]
            for itemx in word1_3[0]:
                W1.append(self.buildFinalVector(mtopic,nword,ntopic,word1,itemx,[0,0]))
            for itemx in word2_3[0]:
                W2.append(self.buildFinalVector(mtopic,nword,ntopic,word2,itemx,[0,0]))
            #W1.append(self.buildFinalVector(mtopic,nword,ntopic,word1,0,word1_3[1]))
            #W2.append(self.buildFinalVector(mtopic,nword,ntopic,word2,0,word2_3[1]))
            avg_sim=0
            for item1 in range(len(W1)):
                for item2 in range(len(W2)):
                    avg_sim+=self.P[word1+'_'+str(2*item)][item1]*self.P[word2+'_'+str(2*item+1)][item2]*self.similarity(W1[item1],W2[item2])
            self.avg_sim_total.append(avg_sim)
            file=open('./final_avg.txt','a')
            file.write(str(avg_sim)+' '+str(self.collection[item][7])+' '+str(self.collection[item][1])+' '+str(self.collection[item][3])+'\n')# 7(avg),18(max),19(min)
            file.close()    
            print('Iteration %d completed!'%item)  
        print('AVGSim completed!')  
        return self.avg_sim_total

    def spearman(self):
        file=open('./scaled.avg.txt','w')
        file1=open('./transformed.txt','w')
        x=[]
        y=[]
        w1=[]
        w2=[]
        for line in open('./final_avg.txt'):
            l=line.split()
            x.append(float(l[0]))
            y.append(float(l[1]))
            w1.append(l[2])
            w2.append(l[3])
        #normalize
        tt=np.abs(x)
        min=np.min(tt)
        x=[l/min for l in x]
        for i in range(len(x)):
            file.write(str(x[i])+' '+str(y[i])+' '+w1[i]+' '+w2[i]+'\n')
        file.close()
        min=np.min(x)
        max=np.max(x)
        x=[(nl-min)/(max-min) for nl in x]
        for i in range(len(x)):
            file1.write(str(x[i])+' '+str(y[i])+' '+w1[i]+' '+w2[i]+'\n')
        file1.close()
        rho,pval= stats.spearmanr(x,y)
        otherinfo=' no noise and abs(Pr)'
        file=open('./final_result_spearmanr.txt','a')
        file.write('pho is '+str(rho)+' pval is '+str(pval)+' finish time '+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time.time()))+' top '+str(self.top)+ otherinfo+'\n')
        file.close()
if __name__=='__main__':
    pp=preprocessing()
    #pp.spearman()
    pp.resolve()
    pp.readNewWordMap()
    pp.readWordvector()
    pp.readTopicvector()
    #pp.Modify()
    pp.build_test_pair()
    pp.find_valid_pair()#word_pair
    pp.extractPwt()
    #a=PWT[0][10000:-1]
    #print(PWT[0][-1])
    pp.extractPtd()
    #a=[[1,2,3,4,5,6,7,8,9,10],[2,4,6,8,10,12,14,16,18,100]]
    #aa=dict()
    #aa['sky']=[1,2,3,4,5,6,7,8,9,10]
    #aa['cloud']=[2,4,6,8,10,12,14,16,18,100]
    #b=pp.normalize_topic(a)
    #c=pp.normalize_word(aa)
    mtopic=pp.magCalc(pp.topicvector_collection)
    ntopic=pp.normalize_topic(pp.topicvector_collection)
    nword=pp.normalize_word(pp.wordvector_dict)
    #pp.topicvector_collection=None
    #pp.wordvector_dict=None
    #tt1=pp.buildFinalVector(mtopic,nword,ntopic,'abus',0,[0,0])
    #tt2=pp.buildFinalVector(mtopic,nword,ntopic,'woodi',0,[0,0])
    #abus3=pp.pzIwc(0,'abus',topic_range=100)
    #woddi3=pp.pzIwc(1,'woodi',topic_range=100)
    pp.avg_calc(mtopic,nword,ntopic)
    pp.spearman()
    #writeColumn(topic_vector,0)
    print('processing finished')