import re
import sys
import codecs
import datetime
import copy
import os

from crf_training import transform_excel2xml
from crf_training import run_parserator
from crf_training import crf_on_dataset

from auto_cluster import entity2vector
from vclusters import vector_clusters

#https://github.com/Travtus/NLP-Analysis/subscription
#0v1# JC Feb 23, 2017  Complete run through

def feb23_main():
    b=[]
    b+=['excel2xml']#ok
    b+=['crf_training']#ok
    b+=['crf_on_dataset']#ok

    b=[] #Resume as memory intense
    b+=['vectorize']
    b+=['cluster']

    b=[] #Debug
    b+=['vectorize']
    
    #########################################
    #  Formal files
    #########################################
    crf_base_name="sink_parser"
    crf_config_file=crf_base_name+"/__init__.py"

    excel_input_filename='./files/corpus1_labelled.csv' #Manually trained doc convert to CRF xml format
    xml_training_filename='./files/corpus1_labelled.xml'
    dataset_input='./files/Cleaned Data.tsv.txt'
    dataset_output_labelled='./files/sample_v1.tsv'
    vectors_filename='entity2vector.vec'

    #########################################
    #  Reduced pipeline run options (fast)
    #########################################
    dataset_label_limit_count=200
    only_cluster_types=['object'] #or [] for all
    
    #1/  Validate imports
    #2/  Transform raw data
    #3/  Pre-clean
    #4/  Training data ok
    #5/  Train CRF
    #    - output intermediary file: ______.
    #7/  Vectorize entities
    #8/  Cluster vectors
    #9/  Discover Root Concepts
    #10/ Output final file: ______.


    #1/  Validate imports
    #2/  Transform raw data
    #[] manually save xmls to csv
    if 'excel2xml' in b:
        transform_excel2xml(filename=excel_input_filename,filename_out=xml_training_filename)
    
    #3/  Pre-clean
    #4/  Training data ok
    #5/  Train CRF
    #    - (recall original parserator init)
    if 'crf_training' in b:
        print "Training crf using config: "+crf_config_file
        print "Training crf using training xml: "+xml_training_filename
        if os.path.exists(crf_config_file) and os.path.exists(xml_training_filename):
            training_ok=run_parserator(crf_base_name,xml_training_filename)
        else:hard_fail=bad_config
    
    
    #6/  Run CRF on raw data
    #    - output intermediary file: ______.
    if 'crf_on_dataset' in b:
        print "Running crf on dataset file: "+dataset_input
        crf_on_dataset(crf_base_name,dataset_input,dataset_output_labelled,dataset_label_limit_count=dataset_label_limit_count)
        print "Dataset labelled as: "+dataset_output_labelled
    
    #7/  Vectorize entities
    if 'vectorize' in b:
        print "Running vectorizer"
        entity2vector(vectors_filename=vectors_filename)

    #8/  Cluster vectors
    if 'cluster' in b:
        print "Running cluster"
        vector_clusters(vectors_filename,only_cluster_types=only_cluster_type)
    
    #9/  Discover Root Concepts
    #10/ Output final file: ______.
    
    return


def main():
    branches=['feb23_main']
    if 'feb23_main' in branches:
        feb23_main()
    return


if __name__=='__main__':
    main()
    
    