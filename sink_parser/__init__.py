#!/usr/bin/python
# -*- coding: utf-8 -*-
import pycrfsuite
import os
import re
import warnings
import string
from collections import OrderedDict


#0v2# JC Feb 8, 2017

#  _____________________
# |1. CONFIGURE LABELS! |
# |_____________________| 
#LABELS = [] # The labels should be a list of strings
LABELS=['noise', 'location', 'object', 'description', 'action', 'numerator', 'conjunction', 'analysis']


#***************** OPTIONAL CONFIG ***************************************************
PARENT_LABEL  = 'TokenSequence'               # the XML tag for each labeled string #name
GROUP_LABEL   = 'Collection'                  # the XML tag for a group of strings #NameCollection
NULL_LABEL    = 'Null'                        # the null XML tag
MODEL_FILE    = 'learned_settings.crfsuite'   # filename for the crfsuite settings file
#************************************************************************************



try :
    TAGGER = pycrfsuite.Tagger()
    TAGGER.open(os.path.split(os.path.abspath(__file__))[0]+'/'+MODEL_FILE)
except IOError :
    TAGGER = None
    warnings.warn('You must train the model (parserator train [traindata] [modulename]) to create the %s file before you can use the parse and tag methods' %MODEL_FILE)

def parse(raw_string):
    if not TAGGER:
        raise IOError('\nMISSING MODEL FILE: %s\nYou must train the model before you can use the parse and tag methods\nTo train the model annd create the model file, run:\nparserator train [traindata] [modulename]' %MODEL_FILE)

    tokens = tokenize(raw_string)
    if not tokens :
        return []

    features = tokens2features(tokens)

    tags = TAGGER.tag(features)
    return list(zip(tokens, tags))

def tag(raw_string) :
    tagged = OrderedDict()
    for token, label in parse(raw_string) :
        tagged.setdefault(label, []).append(token)

    for token in tagged :
        component = ' '.join(tagged[token])
        component = component.strip(' ,;')
        tagged[token] = component

    return tagged


#  _____________________
# |2. CONFIGURE TOKENS! |
# |_____________________| 
#     (\__/) || 
#     (•ㅅ•) || 
def tokenize(raw_string):
    # this determines how any given string is split into its tokens
    # handle any punctuation you want to split on, as well as any punctuation to capture

    if isinstance(raw_string, bytes):
        try:
            raw_string = str(raw_string, encoding='utf-8')
        except:
            raw_string = str(raw_string)
    
    #re_tokens = # re.compile( [REGEX HERE], re.VERBOSE | re.UNICODE)
    #{people source}
    #there is a test file for testing the performance of the tokenize function. You can adapt it to your 
    #needs & run the test w/ nosetests . to ensure that you are splitting strings properly.
    re_tokens = re.compile(r"""
    \bc/o\b
    |
    [("']*\b[^\s\/,;#&()]+\b[.,;:'")]* # ['a-b. cd,ef- '] -> ['a-b.', 'cd,', 'ef']
    |
    [#&@/]
    """,
                           re.I | re.VERBOSE | re.UNICODE)

#address    re_tokens = re.compile(r"""
#address    \(*\b[^\s,;#&()]+[.,;)\n]*   # ['ab. cd,ef '] -> ['ab.', 'cd,', 'ef']
#address    |
#address    [#&]                       # [^'#abc'] -> ['#']
#address    """,
#address                           re.VERBOSE | re.UNICODE)

    #JC Dec 20, 2016#
    #Custom logic 1:  13cb to 13 cb.  Put raw space between any number and character
    while (re.search(r'\d[a-zA-Z]',raw_string)):
        raw_string=re.sub(r'(\d)([a-zA-Z])',r'\1 \2',raw_string)
       
    tokens = re_tokens.findall(raw_string)

    if not tokens :
        return []
    return tokens


#  _______________________
# |3. CONFIGURE FEATURES! |
# |_______________________| 
#     (\__/) || 
#     (•ㅅ•) || 
def tokens2features(tokens):
    # this should call tokenFeatures to get features for individual tokens,
    # as well as define any features that are dependent upon tokens before/after
    
    feature_sequence = [tokenFeatures(tokens[0])]
    previous_features = feature_sequence[-1].copy()

    for token in tokens[1:] :
        # set features for individual tokens (calling tokenFeatures)
        token_features = tokenFeatures(token)
        current_features = token_features.copy()

        # features for the features of adjacent tokens
        feature_sequence[-1]['next'] = current_features
        token_features['previous'] = previous_features        
        
        # DEFINE ANY OTHER FEATURES THAT ARE DEPENDENT UPON TOKENS BEFORE/AFTER
        # for example, a feature for whether a certain character has appeared previously in the token sequence
        
        feature_sequence.append(token_features)
        previous_features = current_features

    if len(feature_sequence) > 1 :
        # these are features for the tokens at the beginning and end of a string
        feature_sequence[0]['rawstring.start'] = True
        feature_sequence[-1]['rawstring.end'] = True
        feature_sequence[1]['previous']['rawstring.start'] = True
        feature_sequence[-2]['next']['rawstring.end'] = True

    else : 
        # a singleton feature, for if there is only one token in a string
        feature_sequence[0]['singleton'] = True

    return feature_sequence


#GLOBAL VARIABLE FEATURES
VOWELS_Y = tuple('aeiouy')
PREPOSITIONS={'aboard', 'about', 'above', 'across', 'after', 'against', 'along', 'amid', 'among', 'anti', 'around', 'as', 'at', 'before', 'behind', 'below', 'beneath', 'beside', 'besides', 'between', 'beyond', 'but', 'by', 'concerning', 'considering', 'despite', 'down', 'during', 'except', 'excepting', 'excluding', 'following', 'for', 'from', 'in', 'inside', 'into', 'like', 'minus', 'near', 'of', 'off', 'on', 'onto', 'opposite', 'outside', 'over', 'past', 'per', 'plus', 'regarding', 'round', 'save', 'since', 'than', 'through', 'to', 'toward', 'towards', 'under', 'underneath', 'unlike', 'until', 'up', 'upon', 'versus', 'via', 'with', 'within', 'without'}

def tokenFeatures(token) :
    # this defines a dict of features for an individual token
    if token in (u'&') :
        token_clean = token_abbrev = token
        
    else :
        token_clean = re.sub(r'(^[\W]*)|([^.\w]*$)', u'', token.lower())
        token_abbrev = re.sub(r'\W', u'', token_clean)
        

    features = {   # DEFINE FEATURES HERE. some examples:
                    'nouns_count' : is_noun(token),
                    'verbs_count': is_verb(token),
                    'nltk_tag': get_tag(token),

                    'is_location': is_location(token),
                    'is_object': is_object(token),

                    'length': len(token),

                    'has.vowels'  : bool(set(token_abbrev[1:]) & set(VOWELS_Y)),
                    'period' : '\.' in token,
                    'comma'  : token.endswith(','), 
                    'prepositions' : token_abbrev in PREPOSITIONS,
                    'possessive' : token_clean.endswith("'s") 
                }
    
#    #FEATURE OPTIONS
#                    'length': len(token),
#                    'case'  : casing(token),
#                    'comma'  : token.endswith(','), 
#                    'hyphenated' : '-' in token_clean,
#                    'bracketed' : bool(re.match(r'(["(\']\w+)|(\w+[")\'])', token) and not re.match(r'["(\']\w+[")\']', token)),
#                    'fullbracketed' : bool(re.match(r'["(\']\w+[")\']', token)),
#                    'contracted' : "'" in token_clean,
#                    'has.vowels'  : bool(set(token_abbrev[1:]) & set(VOWELS_Y)),
#                    'digits' : digits(token_abbrev),
#                    'prepositions' : token_abbrev in PREPOSITIONS,
#                    'trailing.zeros' : (trailingZeros(token_abbrev)
#                                    if token_abbrev.isdigit()
#                                    else False),
#people
#                    'nopunc' : token_abbrev,
#                    'abbrev' : token_clean.endswith('.'),
#                    'initial' : len(token_abbrev) == 1 and token_abbrev.isalpha(),
#                    'just.letters' : token_abbrev.isalpha(),
#                    'roman' : set('xvi').issuperset(token_abbrev),
#                    'endswith.vowel' : token_abbrev.endswith(VOWELS_Y),
#                    'metaphone1' : metaphone[0],
#                    'metaphone2' : metaphone[1],
#                    'more.vowels' : vowelRatio(token_abbrev),
#                    'in.names' : token_abbrev.upper() in ratios,
#                    'first.name' : ratios.get(token_abbrev.upper(), 0),
#                    'gender_ratio' : gender_names.get(token_abbrev, False),
#                    'possessive' : token_clean.endswith("'s") 
#    address
#        features = {'abbrev' : token_clean[-1] == u'.',
#                    'digits' : digits(token_clean),
#                    'word' : (token_abbrev 
#                              if not token_abbrev.isdigit()
#                              else False),
#                    'length' : (u'd:' + str(len(token_abbrev))
#                                if token_abbrev.isdigit()
#                                else u'w:' + str(len(token_abbrev))),
#                    'endsinpunc' : (token[-1]
#                                    if bool(re.match('.+[^.\w]', token, flags=re.UNICODE))
#                                    else False),
#                    'directional' : token_abbrev in DIRECTIONS,
#                    'street_name' : token_abbrev in STREET_NAMES,
#                    'has.vowels'  : bool(set(token_abbrev[1:]) & set('aeiou')),
#                    }
    return features



# define any other methods for features. this is an example to get the casing of a token
def trailingZeros(token) :
    results = re.findall(r'(0+)$', token)
    if results :
        return results[0]
    else :
        return ''
    
def digits(token) :
    if token.isdigit() :
        return 'all_digits' 
    elif set(token) & set(string.digits) :
        return 'some_digits' 
    else :
        return 'no_digits'

def casing(token) :
    if token.isupper() :
        return 'upper'
    elif token.islower() :
        return 'lower' 
    elif token.istitle() :
        return 'title'
    elif token.isalpha() :
        return 'mixed'
    else :
        return False


# Custom features
###############################################################
#http://nbviewer.jupyter.org/github/tpeng/python-crfsuite/blob/master/examples/CoNLL%202002.ipynb

from singfreq import get_noun_verbs
from singfreq import tag_blob

def is_noun(token):
    nouns,verbs,desc=get_noun_verbs(token)
    return len(nouns)
def is_verb(token):
    nouns,verbs,desc=get_noun_verbs(token)
    return len(verbs)
def get_tag(token):
    #please ** tagged as NN if just work.
    #ADV    adverb    really, already, still, early, now
    #TAGS: [('please', 'NN')] for: please
    tag=""
    tags=tag_blob(token)
    for item in tags:
        tag=item[1]
    return tag

#def is_adverb(token):
#    #please ** tagged as NN if just work.
#    #ADV    adverb    really, already, still, early, now
#    tags=tag_blob(token)
#    for item in tags:
#        if item[1]=='ADV':
#            print "YEs adverb: "+token
#            return 1
#    print "TAGS: "+str(tags)+" for: "+token
#    return 0

LOCATIONS=['kitchen','office','deck','landscaping','front','bedroom','touch','apartment','window','bedrooms','community','basement','hall','bank','areas','lanscape','h','l','garage','tower','side','back','escape','7th','living','closet','hallways','outside','exit','c19h','27a','mailbox','courtyard','kitcnen','kitchen','water','bdrms','landscapes','turn','circuit','grounds','garbage','wall','apt','bath','bthrm','1st','unit','bathroom','area','elevator','halls','entrance','door','line','tank','room','roof','shower','r','bdrm','35th','toilet','bathrooms','look','ceiling','wal ks','balcony','livingroom','floor','dining','hot','electrical','stock','livingr','hallway','laundry','staircase','pathways','stairwell','coming','backyard','fl','lobby','building','lawn','light','clean','4e']
OBJECTS=['heater','stove','rod','spreadsheet','radiators','debris','oven','signs','dish','fixtures','toilets','c','aerator','microw','window','combo','weather','blinds','knob','tiles','handle','radiator','intercom','cover','ceil','units','walls','dryer','fan','outlet','front','bar','sill','conditi on','vanity','towel','sleeve','hinges','list','holes','backsplash','micro','egress','trap','pane','fence','frame','fixture','seat','flush','cold','heaters','flower','closet','pictures','pipes','valve','burner','microwave','lights','medicine','washer','moldings','sheetrock','screen','water','sink' ,'key','carbon','tub','roller','kitchen','on','faucet','hinge','range','smoke','head','repair','garbage','wall','vent','ac','bath','down','guard','floors','appliances','unit','bulbs','top','system','elevator','cords','panel','a','ceiling','door','detecto','shelf','gas','dishwasher','cabinet','glass','broke','bulb','sign','holder','basin','grill','waste','tank','look','air','shower','sour','bathtub','toilet','cylinder','lock','entrance','covers','as','bend','exterior','gasket','conditioner','tiling','showerhead','countertop','cabinets','court','monoxide','floor','drains','wiring','snow',' freezer','machine','guards','refridgerator','electrical','inventory','doors','tile','detector','wires','electric','living','outlets','fridge','boiler','coming','device','plugs','refrigerator','sprinkler','building','drain','windows','light','alarm','metal','boxes','pipe','switch','compactor','breaker','tubes','wind']
def is_location(token):
    if len(token)>2:
        if token in LOCATIONS:return 1
    return 0
def is_object(token):
    if len(token)>2:
        if token in OBJECTS:return 1
    return 0



















