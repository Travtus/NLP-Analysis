import re
import sys
import codecs
import datetime
import copy
import random
import string
import time
import numpy as np

import gensim
from gensim.models import Word2Vec

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#If an array is too large to be printed, NumPy automatically skips the central part of the array and only prints the corners:
np.set_printoptions(threshold='nan')

STOPWORDS = set(stopwords.words('english'))

class Word2vec_Model(object):
    #most_similar, doesnt_match, similarity, 
    def __init__(self):
        self.word_tfidf=''
        return
    
    def initialize(self):
        #allocation 3Mx300 floats needs contiguous unfragmented memory >4GB likely 8
        #self.model = Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)  # C binary format
        self.model = Word2Vec.load_word2vec_format('./files/GoogleNews-vectors-negative300.bin', binary=True)  # C binary format
        self.model.init_sims(replace=True) #replace=True
        return

    def set_tfidf_weights(self,word_tfidf):
        self.word_tfidf=word_tfidf
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        self.max_tfidf={}
        for the_type in self.word_tfidf:
            self.max_tfidf[the_type]=0
        for the_type in self.word_tfidf:
            for w in self.word_tfidf[the_type]:#.vocabulary_.items():
                if self.word_tfidf[the_type][w]>self.max_tfidf[the_type]:
                    self.max_tfidf[the_type]=self.word_tfidf[the_type][w]
        return

    def get_tfidf_weight(self,the_type,word):
        w=1
        if self.word_tfidf:
            try:
                w=self.word_tfidf[the_type][word]
            except:
                w=self.max_tfidf[the_type] #max ie/ rare
        return w
    
    def terms2vector(self,entity,the_type=''):
        vector=[]
        org_entity=entity.lower()
        #model['computer']  # raw numpy vector of a word
        #array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
        found=False
        while (not found):
            try:
                vector=self.model[entity]*self.get_tfidf_weight(the_type,entity)
                found=True
            except: pass #quiet
            
            if not found:
            #Do mean vector
                vector=self.sent2vect(entity,the_type=the_type)

                if not np.any(vector):
                    #Final fall-back do entity reduction
                    if len(re.split(r' ',entity))==1:
                        break
                    print "Doing entity reduction for (no mean vectors): "+entity
                    new_entity=self.entity_reduction(entity)
                    if new_entity==entity:
                        break
                    else:
                        entity=new_entity
                else:
                    #D# print "Did mean vector for: "+entity
                    found=True

        #print "GOT: "+str(vector)
        #print "GOT type: "+str(type(vector))
        return vector

    def terms2meanvector(self,terms,dim=300,the_type=''):
        #according to Kenter et al. 2016, "simply averaging word embeddings of all words in a text has proven to be a strong baseline
        #or feature across a multitude of tasks", such as short text similarity tasks.
            #TF-idf weightings option http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec
    
        if False: #nontfidf
            return np.array(
                np.mean([self.model[w] for w in terms if w in self.model]
                        or [np.zeros(dim)], axis=0)
            )
        #def get_tfidf_weight(self,the_type,word):
        return np.array(
            np.mean([self.model[w]*self.get_tfidf_weight(the_type,w) for w in terms if w in self.model]
                    or [np.zeros(dim)], axis=0)
        )

    def filter_stopwords(self,words):
        return [i for i in words if i not in STOPWORDS]

    def sent2vect(self,sentence,the_type=''):
        # http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec
        terms=re.split(r' ',sentence)
        terms=self.filter_stopwords(terms)
        mvector=self.terms2meanvector(terms,the_type=the_type)
        return mvector
    
    def entity_reduction(self,entity):
        #Logic:
        #- ie if no vector found
        #- remove erroneous terms one by one
        #model.doesnt_match("breakfast cereal dinner lunch".split()) cereal
        new_entity=entity
        fields=entity.split()
        if len(fields)>1:
#D            print "Doesn't match: "+str(fields)
            try: bad_field=self.model.doesnt_match(fields)
            except:bad_field=''
            if bad_field:
                #Remove first option of bad
                fields.remove(bad_field)
                new_entity=" ".join(fields)
#D                print "Entity reduced. "+str(entity)+" --> "+str(new_entity)

        if new_entity==entity:
            if len(fields)>1:
                last=fields.pop()
                new_entity=" ".join(fields)
#D                print "Could not semantically reduce. So remove last...first?? term: "+str(entity)+" --> "+str(new_entity)
        return new_entity
     

def load_entities():
    the_list=[]
    fp=open('./files/sample_vA.tsv','r')
    c_location=4
    c_object=5
    c_action=6
    c_numerator=7
    c_analysis=8
    c_description=9 #sentence
    
    entity_lookup=[]
    entity_lookup.append(('location',4))
    entity_lookup.append(('object',5))
    entity_lookup.append(('action',6))
    entity_lookup.append(('numerator',7))
    entity_lookup.append(('analysis',8))
    entity_lookup.append(('description',9))
    
    c=0
    for line in fp.readlines():
        c+=1
        if c==1:continue

        line=re.sub(r'\n','',line)
        fields=re.split(r'\t',line)
        
        for name,col in entity_lookup:
            vv=fields[col]
            vv=vv.strip()
            if vv:
                the_list.append((name,vv))
    return the_list

def generate_TFIDFS(the_list):
    corpus={}
    word_tfidf={}
    for the_type,entity in the_list:
        try:
            corpus[the_type].append(entity)
        except:
            corpus[the_type]=[]
            corpus[the_type].append(entity)

    for the_type in corpus:
        tfidf = TfidfVectorizer()
        tfidf.fit_transform(corpus[the_type])
        word_tfidf[the_type]= dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    return word_tfidf


def entity2vector(vectors_filename='entity2vector'):

    the_list=load_entities()
    print "Resolving "+str(len(the_list))+" entities"
    word_tfidf=generate_TFIDFS(the_list)

    W2V=Word2vec_Model()
    W2V.set_tfidf_weights(word_tfidf)
    print "Loading big GoogleNews model..."
    W2V.initialize()
    print "done."

    saw_these={}
    fp=open('entity2vector','w')
    c=0
    for the_type,entity in the_list:
        c+=1
        if not c%10000: print "Progress: "+str(c)+"/"+str(len(the_list))
        if entity in saw_these:continue
        saw_these[entity]=True

        vector=W2V.terms2vector(entity,the_type=the_type)
        if the_type=="" or entity=="":
            print "Bad type or entity def at list line: "+str(c)
        elif not vector==[]:
            fp.write(the_type+"\t"+entity+"\t"+str(vector)+"\n")
        else:
            print "No vector for: "+str(entity)
    fp.close()
    print "Done."
    return


def terms2meanvector(model, X,dim=300):
    #TF-idf weightings http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec
    return np.array([
        np.mean([model[w] for w in words if w in model]
                or [np.zeros(dim)], axis=0)
        for words in X
    ])

def filter_stopwords(words):
    return [i for i in words.lower() if i not in stop]
        
def terms2vector():
    W2V=Word2vec_Model()
    print "Loading big GoogleNews model..."
    W2V.initialize()
    print "done."

    sentence="connected hose to"
    terms=re.split(r' ',sentence)
    terms=filter_stopwords(terms)
    vector=terms2meanvector(W2V.model, terms)
    print "Sentence: "+str(sentence)
    print "Mean vector: "+str(vector)
    return

def main():
    branches=['terms2vector'] #dev (see w2v class)
    branches=['entity2vector']

    if 'terms2vector' in branches:
        terms2vector()

    if 'entity2vector' in branches:
        entity2vector()
    return

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
