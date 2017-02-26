import re
import sys
import codecs
import datetime
import copy
import os
from collections import OrderedDict

from crf_training import transform_excel2xml
from crf_training import run_parserator
from crf_training import crf_on_dataset

from auto_cluster import entity2vector
from vclusters import vector_clusters

import xml.etree.ElementTree as ET

from autocorrect import spell  #pip install autocorrect

#https://github.com/Travtus/NLP-Analysis/subscription
#0v1# JC Feb 23, 2017  Complete run through

def iter_xml(filename):
    #Notes:
    #- xml tags not iterating friendily

    tree = ET.parse(filename)
    root = tree.getroot()
    ss='{urn:schemas-microsoft-com:office:spreadsheet}'

    fields=[]
    c=0
    for elt in root.iter(ss+'Row'):
        for cell in elt.iter(ss+'Cell'):
            for data in cell.iter():
#D                print "> "+str(data.text)
                fields.append(data.text) #not Number type
        c+=1
        if c>1: #header
            yield fields
        fields=[]
#D        print "------------------------"
    return

def safetext(text):
    try:text=str(text)
    except:pass
    try:text=text.encode('ascii','ignore')
    except:pass
    return text

#LOCAL NEW TRANSFORMS
def transform_rawxml2tsv(xml_dataset_input,dataset_output):
    #Feb 24
    #Given xml, transform to expected tsv format (like Cleaned Data.tsv.txt)
#    fp=open(xml_dataset_input,'')
    #   <Row>
    #    <Cell><Data ss:Type="String">id</Data></Cell>
    #    <Cell><Data ss:Type="String">created_date</Data></Cell>
    #    <Cell><Data ss:Type="String">company_id</Data></Cell>
    #    <Cell><Data ss:Type="String">job_order_number</Data></Cell>
    #7   <Cell><Data ss:Type="String">portfolio_name</Data></Cell>
    #8   <Cell><Data ss:Type="String">address</Data></Cell>
    #9   <Cell><Data ss:Type="String">unit</Data></Cell>
    #10    <Cell><Data ss:Type="String">category</Data></Cell>
    #11    <Cell><Data ss:Type="String">description</Data></Cell>
    #   </Row>
    fp=open(dataset_output,'w')
    fp.write("address\tcategory\tdescription\tunit\n")
    c=0
    for row in iter_xml(xml_dataset_input):
        if row:
#            if True:
            try:
                unit=''
                id=safetext(row[0])
                address=safetext(row[8])
                category=safetext(row[10])

                category=safetext(row[9])
                description=safetext(row[11])
                description=preclean_blob(description)

                liner=address+"\t"+category+"\t"+description+"\t"+unit
                fp.write(liner+"\n")
                c+=1
                pass
            except:
                print "SKIP raw row: "+str(row)
    print "Translated xml rows: "+str(c)+" to: "+dataset_output
    fp.close()
    return

def spell_checker(blob):
    #[] option to train further
    #http://norvig.com/spell-correct.html
    #https://pypi.python.org/pypi/autocorrect/0.1.0 (library based on above)
    nblob=""
    blob=re.sub(r'([\"\/\:\[\]\(\)\.\,\'\\])',r' \1 ',blob) #local tokenizer
    blob=re.sub(r'\s+',' ',blob)
    for word in re.split(r' ',blob):
        nword=word
        if not re.search(r'\d',word) and re.search(r'\w',word):
            nword=spell(word)
            if not word==nword:
                print word+" --re-spell--> "+nword
        nblob+=nword+" "
    return nblob

class NestedDict(dict):
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, NestedDict())
    
def auto_map_fix(blob):
    #**Includes similies
    nblob=blob
    MM=OrderedDict()
    fp=open('./files/important_similies.tsv','r')
    for line in fp.readlines():
        line=re.sub(r'\n','',line)
        line=line.lower()
#D        print "GIVEN: "+str(line)
        fields=re.split(r'\t',line)
        target=fields.pop()
        target=re.sub(r'_',' ',target)
        terms=" ".join(fields).strip()
#        print terms+" --> "+target
        if terms:
            MM[terms]=target
    fp.close()
    
    #Local mapping
    MM['MEDICE']="medicine"
    MM['&amp;']="and"
    MM['--']="-"
    MM['A/C']="AC"
    MM['B/R']="BEDROOM"
    MM['L/R']="LIVINGROOM"
    MM['tnx']="thanks"
    MM['Tnk']="thanks"
    MM['tnk']="thanks"
    MM['bedrm']="bedroom"
    MM['Bthrm']="bathroom"
    MM['pls']="please"
    MM['BRKN']="BROKEN"
    MM['LVRM']="LIVINGROOM"
    MM['HALWAY']="HALLWAY"
    MM['Patchup']="Patch up"
    MM['RCYCLING']="RECYCLING"
    MM['pcs']="pieces"

#    MM['V/B']="later" #?
    
    #Sort by key length
    MMs=OrderedDict(sorted(MM.iteritems(), key=lambda x: len(x[0]),reverse=True))
    
    #Do substitution
    for term in MMs:
#D        print term
        if re.search(r'\b'+term+r'\b',nblob):
            nblob=re.sub(r'\b'+term+r'\b',"xxxxxxx"+MMs[term]+"xxxxxxx",nblob) #replace once
    
    nblob=re.sub(r'xxxxxxx','',nblob)
    nblob=re.sub(r'\s+',' ',nblob) #single spaces
    return nblob


def preclean_blob(blob,verbose=False):
    #LOAD TABLES & FILTERS
    cblob=blob
    
    #2\  Mapping table
    cblob=auto_map_fix(cblob)

    #1\  Spell check
    if verbose:
        print "PRE CLEAN: "+str(blob)
        print "[debug] spell checking..."
    cblob=spell_checker(cblob)
    
    #**optional do map again

    if verbose:
        print "POST CLEAN: "+str(cblob)

    return cblob


def feb23_main():
    #1/  Validate imports
    #2/  Transform raw data
    #3/  Pre-clean
    #4/  Training data ok
    #5/  Train CRF
    #7/  Vectorize entities
    #8/  Cluster vectors
    #9/  Discover Root Concepts

    #########################################
    #  Reduced pipeline run options (fast)
    #########################################
    dataset_label_limit_count=200
    only_cluster_types=['object'] #or [] for all
    
    
    b=[]
    b+=['crf_training']#ok
    b+=['crf_on_dataset']#ok

    b=[] #Resume as memory intense
    b+=['vectorize']
    b+=['cluster']

    b=[] #Debug
    b+=['vectorize']
    b+=['excel2xml']#ok
    
    #New training sources validation
    b=[] #Debug
    b+=['raw input xml 2 tsv']#ok

    b=[] #preclean_blob STANDALONE TEST
    b+=['stand_alone_preclean_blob'] #ok

    b=[] #preclean_blob STANDALONE TEST
    b+=['excel2xml']#ok
    

    if True:
        #RUN ALL OPTIONS
        ##########################################
        dataset_label_limit_count=2000000
        only_cluster_types=[] #[] for all
    
        b=[]
#ok        b+=['excel2xml']
        b+=['raw input xml 2 tsv']
        b+=['crf_training']
        b+=['crf_on_dataset'] #takes while
    
    #    b+=['vectorize']      #takes memory
    #    b+=['cluster']
    #    b+=['stand_alone_preclean_blob'] #ok


    
    
    #########################################
    #  Formal files
    #########################################
    crf_base_name="sink_parser"
    crf_config_file=crf_base_name+"/__init__.py"

    excel_input_filename='./files/corpus1_labelled.csv' #Manually trained doc convert to CRF xml format
    xml_training_filename='./files/corpus1_labelled.xml'
    
    xml_dataset_input='./files/NLP Raw Data Full.xml'
#old#    dataset_input='./files/Cleaned Data.tsv.txt'
    dataset_input='./files/NLP Raw Data Full.xml.tsv'
    dataset_output_labelled='./files/sample_v1.tsv'
    vectors_filename='entity2vector.vec'



    #1/  Validate imports
    #2/  Transform raw data
    
    if 'stand_alone_preclean_blob' in b:
        #Stand alone test!
        blob="The quick browwn 23a fox jmps over the lazy livingr and a c bth dawg.water leak"
        blob=preclean_blob(blob)
        print "Post clean blob: "+str(blob)

    #[] manually save xmls to csv
    if 'excel2xml' in b:
        print "Step 1:  Save given xlsx into csv like corpus1_labelled.csv"
        print "Translating excel training sources from: "+excel_input_filename+" to "+xml_training_filename
        transform_excel2xml(filename=excel_input_filename,filename_out=xml_training_filename)

    if 'raw input xml 2 tsv' in b:
        print "Translating raw xml input..."
        print "Transform raw data input xml to consumabble tsv (similar to org Cleaned Data.tsv.txt"
        #xml_dataset_input='./files/NLP Raw Data Full.xml'
        #dataset_input='./files/NLP Raw Data Full.xml.tsv'
        transform_rawxml2tsv(xml_dataset_input,dataset_input)
    
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
    
    
    
    
    
    
    