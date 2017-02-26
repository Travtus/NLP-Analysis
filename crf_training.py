import os
import time
import re
import subprocess
from threading  import Thread
import importlib
from docutils.utils.math.math2html import LineReader

#MODULAR CRF TRAINING
#0v3# JC Feb 23, 2017  Move into run_pipeline.  Call from cluster_run
#0v2# JC Feb 15, 2017  Transform for crf
#0v1# JC Feb 8, 2017

HARD_LABELS=['location', 'object', 'action', 'numerator', 'analysis','description','conjunction','noise']


class NestedDict(dict):
    def __getitem__(self, key):
        if key in self: return self.get(key)
        return self.setdefault(key, NestedDict())
    
def run_command(command):
    # p = sub.Popen(command,stdout=sub.PIPE,stderr=sub.PIPE)
    # output, errors = p.communicate()
    p = os.popen(command,"r")
    while 1:
        line = p.readline()
        if not line: break
        line=re.sub(r'\n','',line) 
        yield line
    return

def transform_excel2xml(filename='corpus1_labelled.csv',filename_out=''):
    global HARD_LABELS
    #large    noise    hole    description    above    description    shower        head    object    due    noise    to    noise    plumbing    description
    #to:
    #
    #<Collection>
    #<TokenSequence><request>Please</request> <request>roll</request> <request>this</request> <request>booking</request> <request>and</request> <request>send</request> <request>me</request> <request>the</request> <request>revised.</request> <request>Also</request> <request>the</request> <request>total</request> <request>is</request> <request>now</request> <count>3</count> <count_item>skids</count_item> <request>/</request> <weight_value>4500</weight_value> <weight_unit>kgs</weight_unit> <request>/</request> <volume_value>2.98</volume_value> <volume_unit>cbm.</volume_unit> <from_pre>Regards</from_pre> <from_name>Josephine</from_name> <noise>To</noise> <noise>quotes@ecuworldwide.us</noise></TokenSequence>
    #</Collection>
    
    #Exception #1
    #remove,action,smoke,,alarm,object,and,conjunction,install,action,combo,,alarm,object,,,,,,,,
    #if blank space after term then assume combo entity so label both as same (need to get final)
#    from cluster_run import preclean_blob
    all_labels=[]
    
    SAMPLES=NestedDict()
    fp=open(filename,'r')
    f_out=open(filename_out,'w')
    f_out.write('<Collection>\n')
    j=0
    for line in fp.readlines():
        j+=1
#        if j>10:a=stop

        line=re.sub('\n','',line)
        
        #Global patch
        line=line.lower()
        line=re.sub(r'anlysis','analysis',line)
        
        
        fields=re.split(r'\,',line)
        
        #Fill in blanks
        #remove,action,air,,condition,,cover,object,install,action,a,,c,object,,,,,,,,
        #['apartment', '', 'carbon', '', 'monoxide', '', 'smoke', '', 'detector', '', 'repl', '', '', '', '', '', '', '', '', '', '', '']
        c=len(fields)
        label=""
        term=True
        first=True
        bad_label_line=[]
        while(c):
            c-=1
            #Get last label (extra commas!)
            if first and fields[c]:
                label=fields[c]
                first=False
                term=True
            elif not first:
                if term:
                    term=False
                else: #label
                    if fields[c]:
                        if not fields[c] in all_labels:
                            all_labels.append(fields[c])
                        if not fields[c] in HARD_LABELS:
                            bad_label_line.append(fields[c])
                    term=True
                    #If should be label. If label blank assign previous
                    if not fields[c]:
                        fields[c]=label
        if bad_label_line:
#            print "Warning bad labels: "+str(bad_label_line)
            print "Bad labels: "+line
            continue
        elif not fields[1]:
            print "Skipping bad training line: "+str(line)
            continue
    
        liner="<TokenSequence>"
        term=True #flipflop
        for i,cell in enumerate(fields):
            #print "FO: "+str(i)+" Length fields: "+str(len(fields))+" here; "+str(fields[i])+" > "+str(fields)
            if term and cell: #else access via index
                term=False
                if not fields[i+1]:
                    print "Bad line: "+line
                    continue
                else:
                    SAMPLES[fields[i+1]][cell]=True #option to count
                    liner+="<"+fields[i+1]+">"+cell+"</"+fields[i+1]+"> "
                #except:pass#silent extra csvs
            else:term=True #flip flop
        liner+="</TokenSequence>"
#D        print "---> "+liner
        f_out.write(liner+"\n")
    f_out.write('</Collection>')
    fp.close()
    f_out.close()
    print "Outputted to: "+filename_out
    
    #Output list for training
    samp="locations=["
    for ss in SAMPLES['location']:
        samp+="'"+ss+"',"
    print samp
    samp="object=["
    for ss in SAMPLES['object']:
        samp+="'"+ss+"',"
    print samp

    return all_labels


def test_model(base_name):
    #module = importlib.import_module(base_name, package=None)
    import sink_parser
    print "Model labels: "+str(sink_parser.TAGGER.labels())
    print "Module loaded"
    
    tests=[]
    tests.append("Please check kitchen light it makes noise")

    for st in tests:
        out=sink_parser.parse(st.lower())
        print st+"--> "+str(out)

        #ok#  out=sink_parser.tag(st.lower())
        #ok#  print st+"--> "+str(out)
    
    # https://python-crfsuite.readthedocs.io/en/latest/pycrfsuite.html
    #err info = sink_parser.TAGGER.info()
    #info = sink_parser.TAGGER.dump()
    

    return

def sample_out(base_name):
    #module = importlib.import_module(base_name, package=None)
    import sink_parser
    print "Parsing sample file (rather long)"
    fin='Cleaned Data template.txt'
    fin='c2.csv'
    fout='sample_v1.txt'
    fp=open(fin,'r')
    f_out=open(fout,'w')

    c=0
    for line in fp.readlines():
        c+=1
        line=re.sub(r'\n','',line)
        if not c%1000:print str(c)+") "+line
#D        print line
        try:
            out=sink_parser.parse(line.lower())
            f_out.write(line+"--> "+str(out)+"\n")
        except:pass

    fp.close()
    f_out.close()
    return

def combine_subsequent_objects(tagged):
    #Combine tags where possible (or multiple?)
    combined_tagged=[]
    #Group subsequent object labels into single entity
    last_label=''
    object=""
    for tag in tagged:
        term=tag[0]
        label=tag[1]
        ok_append=True
        if label==last_label:
            if label=='object':#append to previous
                new_term=combined_tagged[len(combined_tagged)-1][0] #get last term
                new_term+=" "+term
                ok_append=False
                combined_tagged[len(combined_tagged)-1]=(new_term,label)
        if ok_append:
            combined_tagged.append((term,label))
            
        last_label=label
    return combined_tagged

def tagged2columns(labels,tagged):
    #1/  combine subsequent objects into single entity
    #2/  group all other labels into one
    #3/  allow for entity2 column for more then 1 object
    all_labels=labels+['object2']
    
    tagged=combine_subsequent_objects(tagged)
    #Combine tags where possible (or multiple?)
    
    dd={}
    for label in all_labels:
        dd[label]=''

    #Group subsequent object labels into single entity
    last_label=''
    object=""
    for tag in tagged:
        term=tag[0]
        label=tag[1]
        label=re.sub(r'decription','description',label) #patch
        label=re.sub(r'inspect','action',label) #patch
        try: 
            if dd[label] and label=='object':
                label='object2' #Transform extra entities into another object
        except:
            print "[debug] bad label found: "+str(label)
            continue

        if dd[label]: #append as exists
            dd[label]+=" "+term
        else:
            dd[label]=term
    
    #Data list
    all_data=[]
    for label in all_labels:
        all_data.append(dd[label])
    return all_labels,all_data


def crf_on_dataset(base_name,dataset_input,dataset_output_labelled='',dataset_label_limit_count=-1):
    #module = importlib.import_module(base_name, package=None)
    import sink_parser
    print "Parsing sample file (rather long)"
#    fin='Cleaned Data.tsv.txt'
#    fout='sample_v1.tsv'
    fp=open(dataset_input,'r')
    f_out=open(dataset_output_labelled,'w')

    c=0
    for line in fp.readlines():
        c+=1

        line=re.sub(r'\n','',line)
        fields=re.split(r'\t',line)
        
        blob=fields[2]#address,category,description,unit

#D        print blob
        try: tagged=sink_parser.parse(blob.lower())
        except:tagged=[]
        column_headers,column_data=tagged2columns(HARD_LABELS, tagged)
        
        #Prepend original line
        column_headers=['address','category','description','unit']+column_headers
        column_data=[fields[0],fields[1],fields[2],fields[3]]+column_data

        if not c%1000:
            try: print str(c)+") "+str(column_data)
            except:pass#quiet
        
        if c==1:#Header
            f_out.write("\t".join(column_headers)+"\n")
        else:
            #print str(column_data)
            f_out.write("\t".join(column_data)+"\n")
        
        #Early exit (during debug)
        if dataset_label_limit_count>0 and c>dataset_label_limit_count:
            print "[debug] exiting dataset labelling early exit at: "+str(c)
            break

    fp.close()
    f_out.close()
    return
    
def run_parserator(base_name,xml_filename):
    #    xml_filename='output_corpus1_labelledcsv.xml'
    training_ok=False
    if True:
        try: os.remove('./'+base_name+"/learned_settings.crfsuite")
        except:
            print "(no model to overwrite)"
        print "Training model..."
        cmd="parserator train "+xml_filename+" "+base_name
        print cmd
        for line in run_command(cmd):
            #Catch fails
            if re.search(r'Done training',line):training_ok=True
            print line

    return training_ok
    

def main():
    global HARD_LABELS
    a=see_cluster_run_py
    #Validate environment
    base_name='sink_parser'
    b=[]
    b+=['1initiate'] #once or check
    b+=['3set_config']
    b+=['2transform_excel2xml'] #once or redo
    b+=['4run_training']
    b+=['2transform_excel2xml','3set_config','4run_training','5test_model']
    b+=['5test_model']
    b+=['2transform_excel2xml','4run_training','5test_model']
    b+=['4run_training','5test_model']
    b+=['4run_training','sample_out']
    b=[]
    b+=['crf_on_dataset']

    if '1initiate' in b:
        #http://parserator.readthedocs.io/en/latest/
        cmd='parserator init '+base_name
        for line in run_command(cmd): print line
        cmd='python setup.py develop'
        for line in run_command(cmd): print line

    if '2transform_excel2xml' in b:
        transform_excel2xml()
    
    if '3set_config' in b:
        print "Config file: "+base_name+"/__init__.py"
        print "Add these labels: "+str(HARD_LABELS)

    if '4run_training' in b:
        a=see_run_parserator()
        #Force retrain
        try: os.remove('./'+base_name+"/learned_settings.crfsuite")
        except:
            print "(no model to overwrite)"
        print "Training model..."
        xml_filename='output_corpus1_labelledcsv.xml'
        cmd="parserator train "+xml_filename+" "+base_name
        print cmd
        for line in run_command(cmd): print line
        
    if '5test_model' in b:
        test_model(base_name)

    if 'crf_on_dataset' in b:
        crf_on_dataset(base_name)
        
        
    

    return

if __name__=='__main__':
    main()





































