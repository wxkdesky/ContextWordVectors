import numpy as np
import sys
import string
import os
import shutil
from sklearn import svm
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class text_classification:
    word_map_true_id=[]
    document_collection=[]
    mag_topic_collection=[]
    wt_collection=[]
    Ptd_collection=[]
    wordmap_collection=[]
    topicvector_collection=[]
    wordvector_dict=dict()
    tfidf_collection=[]
    total_words=[]
    totalDocs=0
    final_doc_feature_vector=[]
    root='./parameter'
    word_vector_path='./parameter/word_vector.txt'
    topic_vector_path='./parameter/topic_vector.txt'
    def __init__(self):
        print('text_classification begins...')
    
    #i represents the line sequence of the word begins with 0 in wordmap.txt(the first line is ignored),index of the collection is the id of the topic
    def extractWordTopic(self,suffix='./parameter/train/alt.atheism.modified.total.final/alt.atheism.modified.total.final.tassign'):
        for line in open(suffix):           
            item=[]
            for combine in line.split():
                #item.append((i,float(p)))
                item.append(combine)
            self.wt_collection.append(item)
        print('word topic extraction finished!length is '+str(len(self.wt_collection[0])))
        return self.wt_collection

    #i represents the id of the topic,index of collection is the index of doc
    # def extractPtd(self,suffix='./parameter/model-01800.theta'):
    #     for line in open(suffix):
    #         item=[]
    #         for i,p in enumerate(line.split()):
    #             #item.append((i,float(p)))
    #             item.append(float(p))
    #         self.Ptd_collection.append(item)
    #     print('Ptd finished!length is '+str(len(self.Ptd_collection[0])))
    #     return self.Ptd_collection

    #index of readWordMap collection is the i of extractPwt collection
    def readWordMap(self,suffix='./parameter/new_wordmap.txt'):
        count=0;
        for line in open(suffix):
            count+=1
            if count>1:
                self.wordmap_collection.append(line.split()[0])# the true id of the word in the wordmap.txt is discarded cause it can be replaced with index of readWordMap collection
                self.word_map_true_id.append(line.split()[1])
        #$self.wordmap_collection=list(set(self.wordmap_collection))
        print('totoal words in map '+str(len(self.wordmap_collection)))
        return self.wordmap_collection
    
    def loadData(self,path):
        count=0
        for line in open(path):
            words_in_document=dict()
            count+=1
            if count>1:
                tokens=line.split()
                for item in tokens:
                    if item in words_in_document:
                        words_in_document[item][0]+=1
                    else:
                        words_in_document[item]=[1,0.0]
                        #self.total_words.append(item)
                    words_in_document[item][1]=0.0
                self.document_collection.append(words_in_document)
        self.totalDocs=len(self.document_collection)

    def readWordvector(self):
        suffix=self.word_vector_path
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

    def readTopicvector(self):
        suffix=self.word_vector_path
        for line in open(suffix):
            count=0
            item=[]
            for i in line.split():
                count+=1
                if count>1:
                    try:
                        v=float(i)
                        item.append(v)
                    except:
                        print('Error in converting str to float in %s' %suffix)
            self.topicvector_collection.append(item)
            #print(len(item))
        print('topic vector count '+str(len(self.topicvector_collection)))
        return self.topicvector_collection

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

    #compute topic vector
    def magCalc(self,topic):
        for item in topic:
            self.mag_topic_collection.append(np.linalg.norm(item))
            #self.mag_topic_collection.append(1)
        return self.mag_topic_collection

    def tfidf_calc(self):
        DF=dict()
        for word in self.wordmap_collection:
            count=0
            for doc in self.document_collection:
                ll=doc.keys()
                if word in doc.keys():
                    if DF.has_key(word):
                        DF[word]+=1
                        count+=1
                    else:
                        DF[word]=2
                        count=1
            if count==0:
                DF[word]=0
        for doc in self.document_collection:
            for word in doc.keys():
                wfreq=doc[word][0]
                dfreq=DF[word] if DF.has_key(word) else 1
                #va=1+np.log2(wfreq/len(doc))
                length=len(doc)
                tf=float(wfreq)/length
                idf=np.log2(float(self.totalDocs)/dfreq)
                tfidf=tf*idf
                doc[word][1]=tfidf

    def getWordTopic(self):
        self.magCalc(self.topicvector_collection)
        iii=-1;
        for doc in self.document_collection:
            iii+=1
            #print('i=%d'%iii)
            #print('doc length is %d'%len(doc))
            for xword in doc.keys():
                index=-1
                for i in range(len(self.wordmap_collection)):
                    if xword ==self.wordmap_collection[i]:
                        index=i
                if index!=-1:
                    topic=0
                    wt=[]
                    for item in self.wt_collection[iii]:
                        wt.append((item.split(':')[0],item.split(':')[1]))
                    for item in wt:
                        if str(item[0])==str(self.word_map_true_id[index]):
                            topic=int(item[1])
                            break
                    #topic=int(self.wt_collection[iii][int(self.word_map_true_id[index])].split(':')[1])
                    #topic_top=[]
                    #for i in range(len(self.topicvector_collection.count)):
                    #    topic_top.append(self.Pwt_collection[i][index])
                    #topic_top=sorted(topic_top,cmp=None,key=None,reverse=True)
                    original_topic_vector=self.normalize_topic(self.topicvector_collection)[topic]
                    final_topic_vector=[float(x)*self.mag_topic_collection[topic] for x in original_topic_vector]
                    doc[xword].append(final_topic_vector)
                else:
                    doc[xword].append([0 for x in range(400)])
                if xword in self.wordvector_dict.keys():
                    doc[xword].append(self.wordvector_dict[xword])
                else:
                    doc[xword].append([0 for x in range(400)])

    def buildFinalVector(self,write_path):
        for doc in self.document_collection:
            a=np.zeros_like([0 for x in range(800)])
            for word in doc.keys():
                b=doc[word][2]
                b.extend(doc[word][3])
                if len(b)==800:
                    b=[float(x)* doc[word][1] for x in b]
                else:
                    b=b[0:800]
                a=a+np.array(b)
            self.final_doc_feature_vector.append(a.tolist())
        w=open(write_path,'w')
        for item in self.final_doc_feature_vector:
            w.write(str(item))
            w.write('\n')
        w.close()
        print('file %s is created!'%(write_path))

    def testData(self,wordmap,tassign,file,final_file):
        self.readWordMap(suffix=wordmap)
        self.readTopicvector()
        self.readWordvector()
        self.extractWordTopic(suffix=tassign)
        # self.readWordvector()
        # self.readTopicvector()
        # mtopic=self.magCalc(self.topicvector_collection)
        # ntopic=self.normalize_topic(self.topicvector_collection)
        # nword=self.normalize_word(self.wordvector_dict)
        self.loadData(file)
        self.tfidf_calc()
        self.getWordTopic()
        self.buildFinalVector(final_file)
        print('test pause')



def performAllTask(path):
    files=os.listdir(path)
    for file in files:
        if os.path.isdir(path+'/'+file):
            new_dir=path+'/'+file#train
            for xfile in os.listdir(new_dir):
                if os.path.isdir(new_dir+'/'+xfile):
                    new_new_dir=new_dir+'/'+xfile#certain folder
                    tassign='n'
                    f_i_l_e='n'
                    wordmap='n'
                    for xxfile in os.listdir(new_new_dir):#certain file
                        if 'tassign' in xxfile:
                            tassign=new_new_dir+'/'+xxfile
                        a=xxfile[len(xxfile)-5:]
                        if a=='final':
                            f_i_l_e=new_new_dir+'/'+xxfile
                        if 'new_word' in xxfile:
                            wordmap=new_new_dir+'/'+xxfile
                    final_file=f_i_l_e+'.feature'
                    if tassign!='n'and f_i_l_e!='n' and wordmap!='n':
                        print('Processing file %s......'%(new_new_dir))
                        cla=text_classification()
                        cla.testData(wordmap,tassign,f_i_l_e,final_file)
    print('all tasks done!')

if __name__=='__main__':
    #cla=text_classification()
    #wordmap='./parameter/train/alt.atheism.modified.total.final/new_wordmap.txt'
    #word_vector_path='./parameter/word_vector.txt'
    #topic_vector_path='./parameter/topic_vector.txt'
    #tassign='./parameter/train/alt.atheism.modified.total.final/alt.atheism.modified.total.final.tassign'
    #file='./parameter/train/alt.atheism.modified.total.final/alt.atheism.modified.total.final'
    #final_file='./parameter/train/alt.atheism.modified.total.final/alt.atheism.modified.total.final.feature'
    #cla.testData(wordmap,tassign,file,final_file)
    path='./parameter'
    performAllTask(path)