import sys
final=[]
with open("tmp/topic.file",'rb') as topic,open ("tmp/word.file",'rb') as word:
    for line1, line2 in zip(word, topic):
                #line1 = line1.decode('utf8',errors='ignore').encode('utf8')
                #line2 = line2.decode('utf8',errors='ignore').encode('utf8')
                #words = utils.to_unicode(line1).split()
                #topics =utils.to_unicode(line2).split()
                words = line1.strip().split()
                topics = line2.strip().split()
                final.append([(it1, int(it2)) for (it1, it2) in zip(words, topics)])
print("finished")