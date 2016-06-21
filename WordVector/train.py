#!/usr/bin/env python2
#-*- coding: UTF-8 -*-
#File: train.py
#Date: 20160521
#Original Author: Yang Liu <largelymfs@gmail.com>
#Modified by Wishcome
#Description Train the topic representation using the topic model and the word2vec's skip gram

import gensim #modified gensim version
import pre_process # read the wordmap and the tassgin file and create the sentence
import sys
if __name__=="__main__":
    #if len(sys.argv)!=4:
    #    print ("Usage : python train.py wordmap tassign topic_number")
    #    sys.exit(1)	
    reload(sys)
    sys.setdefaultencoding('utf-8')
    wordmapfile = "wordmap.txt"
    tassignfile = "model-01800.tassign"
    topic_number = 400
    #wordmapfile is like this: [number] [word],here is to store word in id2word by [number]
    id2word = pre_process.load_id2word(wordmapfile)
    #tassignfile is like this: [number1]:[number2] where number1 is the word_id and number2 is the topic_id
    #and write topic to topic.file and write word in word.file
    pre_process.load_sentences(tassignfile, id2word)
    #Divide word.file into sentences(one sentence is one document) 
    sentence_word = gensim.models.word2vec.LineSentence("tmp/word.file")
    print ("Training the word vector...")
    w = gensim.models.Word2Vec(sentence_word,size=400, workers=20)
    #organize all words and topics in tuple (word,topic)[just like the .tassign file]
    sentence = gensim.models.word2vec.CombinedSentence("tmp/word.file","tmp/topic.file")
    print ("Training the topic vector...")
    w.train_topic(topic_number, sentence)
    print ("Saving the topic vectors...")
    w.save_topic("output/topic_vector.txt")
    print "Saving the word vectors..."
    w.save_wordvector("output/word_vector.txt")
