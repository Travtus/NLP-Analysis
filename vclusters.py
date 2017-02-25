import re
import sys
import codecs
import datetime
import copy
import random
import string
import time

import numpy as np

from sklearn.cluster import KMeans
#https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-3-more-fun-with-word-vectors

class NestedDict(dict):
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, NestedDict())

def vector_clusters(vectors_filename,only_cluster_types=[]):

    dd=NestedDict()
    #fp=open('e2v.txt','r') #sort|uniq
    fp=open(vectors_filename,'r')
    the_type=''
    count=0
    for line in fp.readlines():
        line=re.sub(r'\n','',line)
        fields=re.split(r'\t',line)
        if re.search(r'\[',line):
            #if the_type and not entity=="____" and not vv=="[]": #not first
            if the_type and ((the_type in only_cluster_types)or(only_cluster_types==[])):
                vv=re.sub(r'[\[\]]','',vv).strip()
                vv_list=re.split(r'\s+',vv)
                vector=np.asarray(vv_list)
                try:
                    vector = vector.astype(np.float64)
                except:
                    print "Could not convert: "+str(vv_list)+" ala: <"+str(vv)+">"
                    a=stop

                if count==0:print "Vector length: "+str(np.linalg.norm(vector))
                dd[the_type][entity]=vector
                count+=1

            the_type=fields[0]
            entity=fields[1]
            vv=fields[2]

        else:
            vv+=line

    print "Loaded "+str(count)+" vectors..."


    # DO CLUSTERING
    for the_type in dd:
        print "Doing clustering for: "+the_type

        #\1  Load type vectors
        word_vectors=[]
        terms=[]
        for entity in dd[the_type]:
            vector=dd[the_type][entity]
            word_vectors.append(vector)
            terms.append(entity)
        terms_list=np.asarray(terms)
        word_vectors=np.asarray(word_vectors) #np array of vectors
    
        start = time.time() # Start time
        
        # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an # average of 5 words per cluster
        num_clusters = word_vectors.shape[0] / 5
        
        #\2  Do clustering
        print "Clustering with k-means count: "+str(num_clusters)
        # Initalize a k-means object and use it to extract centroids
        kmeans_clustering = KMeans( n_clusters = num_clusters )
        idx = kmeans_clustering.fit_predict( word_vectors )
        

        end = time.time()
        elapsed = end - start
        print "Time taken for K Means clustering: ", elapsed, "seconds."
    
        #\3  Output results
        word_map=dict(zip(terms_list,idx))
        f_out=open('./files/cluster_'+the_type+'.txt','w')
        for cluster in xrange(0,num_clusters+1):
            # Find all of the words for that cluster number, and print them out
            words = []
            for i in xrange(0,len(word_map.values())):
                if( word_map.values()[i] == cluster ):
                    words.append(word_map.keys()[i])
            print "\nCluster %d" % cluster
            f_out.write("\nCluster %d" % cluster)
            f_out.write("\n")
            if len(words)>1:
                # Print the cluster number  
                print words
                f_out.write(str(words)+"\n")
        f_out.close()
        
    print "Done"
    return



def main():
    branches=['vector_clusters']
    if 'vector_clusters' in branches:
        vector_clusters()
    return

if __name__=='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
