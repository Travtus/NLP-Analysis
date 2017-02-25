import re
import sys
import codecs
import datetime
import copy
import random
import string
import time
import numpy as np
from nltk import ngrams

import gensim
from gensim.models import Word2Vec

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#0v2# JC Feb 24, 2017  Add n-terms to tfidf and vector lookup

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
    
    def get_vector(self,term):
        #Basic wrap for lookup
        #Note: words_are underscored
        term=re.sub(r' ','_',term)
        try:
            vector=self.model[term]
        except:
            vector=np.zeros(300)#False# or [np.zeros(dim)], axis=0)
        return vector

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
    
    def terms2vector(self,entity,the_type='',verbose=True):

        ################################
        #Main entry
        ################################


        vector=[]
        org_entity=entity.lower()
        #model['computer']  # raw numpy vector of a word
        #array([-0.00449447, -0.00310097,  0.02421786, ...], dtype=float32)
        found=False
        while (not found):
            #Vectorize attempt 1
            #^^^^^^^^^^^^^^^^^^^^^^^
            if np.any(self.get_vector(entity)):
                vector=self.get_vector(entity)*self.get_tfidf_weight(the_type,entity)
                found='direct1'
            
            if not found:
                #Vectorize attempt 2
                #^^^^^^^^^^^^^^^^^^^^^^^
                vector=self.sent2vect(entity,the_type=the_type) #Do mean vector

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
                    found='sentence2'

        if verbose:
            print "Found: "+str(found)+" "+str(entity)

        #print "GOT: "+str(vector)
        #print "GOT type: "+str(type(vector))
        return vector

    def _terms2meanvector(self,terms,dim=300,the_type=''):
        #**Note:
        #according to Kenter et al. 2016, "simply averaging word embeddings of all words in a text has proven to be a strong baseline
        #or feature across a multitude of tasks", such as short text similarity tasks.
            #TF-idf weightings option http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec
        #Assumptions:
        #- note: does double check based on original lambda structure
    
        if not the_type:# Do non-tfidf
            return np.array(
                np.mean([self.get_vector(w) for w in terms if np.any(self.get_vector(w))]
                        or [np.zeros(dim)], axis=0)
            )
        else:
            #def get_tfidf_weight(self,the_type,word):
            return np.array(
                np.mean([self.get_vector(w)*self.get_tfidf_weight(the_type,w) for w in terms if np.any(self.get_vector(w))]
                        or [np.zeros(dim)], axis=0)
            )

    def filter_stopwords(self,words):
        return [i for i in words if i not in STOPWORDS]

    def sent2vect(self,sentence,the_type=''):
        #Notes:
        #- looks for longest n-grams prior to vector mean
        
        # http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec
        
        # Approach 1:  remove stopwords, do mean terms
        if False:
            pass
            #wordonly        terms=re.split(r' ',sentence)
            #wordonly        terms=self.filter_stopwords(terms)
            #wordonly        mvector=self.terms2meanvector(terms,the_type=the_type)
        
        # Approach 2:  look for n-grams within sentence
        mvector=self.sent2grams2vector(sentence,the_type=the_type)
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
    
    ################################
    # More refined sent2gram2vect -- takes into account longest entities
    ################################
    def sentence2ngrams(self,sentence,max_ngrams=4):
        ss=sentence.split()
        cgrams=max_ngrams
        while(cgrams):
            grams=ngrams(ss,cgrams)
            for glist in grams:
                yield " ".join(glist)
            cgrams-=1
        return

    def sent2grams2vector(self,sentence,force_lower=True,flag_filter_stopwords=True,the_type=''):
        #Notes:
        #- the_type for tf-idf lookup if type specified
        
        #Transforms sentence into longest existing n-grams then calcs mean vector
        #Assumption only lower as corpus considered mixed messy
        #New York yes.  new york no.
        if force_lower:
            sentence=sentence.lower() #Does mixed -- but lower for now
    
        active_sentences=[sentence]
        longest_terms=[]
        while (active_sentences):
            sentence=active_sentences.pop(0) #pop off head
            print "--> Doing: "+str(sentence)
    
            #1/ Iterate through longest sequences in sentence until a match
            for terms in self.sentence2ngrams(sentence):
                if flag_filter_stopwords:
                    if terms in self.filter_stopwords(terms):continue
                    
                if np.any(self.get_vector(terms)):
#D#                    print "---has---------------> "+terms
                    longest_terms.append(terms)
                    #Assumptions ***:
                    # - remove terms from active sentence -- splits sentence into 2!
                    # - watch as may match subset! ideally check for boundaries []
                    # - also: removes all instances with split**
                    leftovers=re.split(r''+terms,sentence)
                    for left in leftovers:
                        left=left.strip()
                        if left:
                            active_sentences.append(left)
#D                    print "[debug] sentence found: "+terms+" split into: "+str(leftovers)
                    break #Break out of for loop
                else:
                    pass# print "Model has no: ["+str(terms)+"]"
    
        print "[debug]  Longest entities with vectors: "+str(longest_terms)
        #Do 'words' terms to mean vector where words here is terms
        mvector=self._terms2meanvector(longest_terms,the_type='')
        return mvector
     

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
    ngrams=4
    corpus={}
    word_tfidf={}
    for the_type,entity in the_list:
        try:
            corpus[the_type].append(entity)
        except:
            corpus[the_type]=[]
            corpus[the_type].append(entity)

    print "[debug] tfidf upto "+str(ngrams)+' gram'
    for the_type in corpus:
        tfidf = TfidfVectorizer(ngram_range=(1,ngrams))
        tfidf.fit_transform(corpus[the_type])
        word_tfidf[the_type]= dict(zip(tfidf.get_feature_names(), tfidf.idf_))

    return word_tfidf


def entity2vector(vectors_filename='entity2vector'):

    the_list=load_entities()
    print "Resolving "+str(len(the_list))+" entities"

    print "Generating TFIDFS across types..."
    word_tfidf=generate_TFIDFS(the_list)

    W2V=Word2vec_Model()
    W2V.set_tfidf_weights(word_tfidf)
    print "Loading big GoogleNews model..."
    W2V.initialize()
    print "done word2vec initialization"

    saw_these={}
    fp=open(vectors_filename,'w')
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
    a=old
    return np.array([
        np.mean([model[w] for w in words if w in model]
                or [np.zeros(dim)], axis=0)
        for words in X
    ])

def filter_stopwords(words):
    return [i for i in words.lower() if i not in stop]
        
def terms2vector():
    #**local
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

def gram_iterator(sentence):
    ss=sentence.split()
    max_ngrams=4
    cgrams=max_ngrams
    while(cgrams):
        grams=ngrams(ss,cgrams)
        for glist in grams:
#D            print str(glist)
            yield " ".join(glist)
        cgrams-=1
    return

def sentence2longestngrams(sentence):
    for terms in gram_iterator(sentence):
#D        print terms
        yield terms 
    return



def find_longest_ngram_vectors():
    W2V=Word2vec_Model()
    print "Loading big GoogleNews model..."
    W2V.initialize()
    
    print "model checks:"
    try_these=['new york','new_york','New York','New_York']
    #**only New_York
    for tt in try_these:
        try:
            W2V.model[tt]
            aa=True
        except:
            aa=False
        print tt+"--> "+str(aa)

    #> Kathy_Dehner
    #> anti_Westernism
    #> Eure
    #> NORTHLAND_EXPOSURE_ARTISTS_GALLERY
    #> White_Spunner_Construction
    #> Sandip_Nandy
    #> Bad_Habits
    #> hangs_precariously
    #aaa=100
    #while(aaa):
    #    aaa-=1
    #    for word in W2V.model.vocab:
    #        print "> "+word

    sentence="The New York quick brown fox jumps over the lazy dog"
    mvector=W2V.sent2grams2vector(W2V,sentence)
    return


def main():
    branches=['terms2vector'] #dev (see w2v class)
    branches=['entity2vector']
    branches=['find_longest_ngram_vectors']

    if 'terms2vector' in branches:
        terms2vector()

    if 'entity2vector' in branches:
        entity2vector()

    if 'find_longest_ngram_vectors' in branches:
        find_longest_ngram_vectors()
    return

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
