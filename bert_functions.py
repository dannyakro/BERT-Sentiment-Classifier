from transformers import BertTokenizer, BertForMaskedLM, pipeline
import torch
import string
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
import numpy as np
import datetime
import time 
import spacy
import pandas as pd
import re

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english')) 

"""
Text pre-processing functions
To split those cut words with come with punctuation e.g. "like.", ". enable"
"""

Gelphi_output_file_path = "../source_data/gephi_output_cleaned.csv" 

network = pd.read_csv(Gelphi_output_file_path)
network.set_index(network['Label'],inplace=True)

"""
Two models are available, one is uncased, the other one is cased, 
Change according to the importance of CASE in the sentence
- bert-base-cased: Model is case-sensitive and there is a difference between 'english' and 'English'
- bert-base-uncased: Model is case-insensitive and there is no difference between 'english' and 'English'
"""

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased').eval()

### Get the predicted words from  BERT model
def decode(tokenizer, pred_idx, top_clean=5):
    """
    ignore_tokens consists of punctuations and padding tokens
    which are denoted by [PAD].

    Parameters
    ----------
    tokenizer: BertTokenizer Model,
        The loaded tokenizer - can be either bert-base-cased or bert-base-uncased

    pred_idx: Tensor object,
        The tensor object that contains the decoded words

    top_clean: int,
        The number of predicted words returned from the BERT model. Default is 5.

    Returns
    -------
    Returns a list tokens where the top n predicted words are returned, in which
    n is denoted by top_clean
    """
    ignore_tokens = string.punctuation + '[PAD]'
    tokens = []
    for w in pred_idx:
        token = ''.join(tokenizer.decode(w).split())
        if token not in ignore_tokens:
            tokens.append(token.replace('##', ''))
    return tokens[:top_clean]  ## each line one prediction

### Encode the words by BERT tokenizers
def encode(tokenizer, text_sentence, add_special_tokens=True):
    """
    Replace <mask> with the bert tokenizer's mask token which is denoted by <MASK>.

    Parameters
    ----------
    tokenizer: BertTokenizer Model,
        The loaded tokenizer - can be either bert-base-cased or bert-base-uncased

    text_sentence: str,
        The input sentence for the document

    add_special_tokens: boolean,
        Boolean on whether or not to encode the sequences with special tokens
        relative to their model.

    Returns
    -------
        Returns two tensor objects after calling the BERT tokenizer encoder, one with the input IDs
        and the other with the masked word index.
    """
    text_sentence = text_sentence.replace('<mask>', tokenizer.mask_token)
    # if <mask> is the last token, append a "." so that models dont predict punctuation.
    if tokenizer.mask_token == text_sentence.split()[-1]:
        text_sentence += ' .'
    input_ids = torch.tensor([tokenizer.encode(text_sentence, add_special_tokens=add_special_tokens)])
    mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]
    return input_ids, mask_idx

###
def get_predictions(text_sentence,top_clean=5):
    """
    Runs the bert model with Pytorch's disabled gradient calculation
    
    Parameters
    ---------
    text_sentence: str,
        The input sentence for the document

    top_clean: int,
        The number of predicted words returned from the BERT model. Default is 5

    Returns
    -------
    list
        List of predicted words after decoding the masked word
    """
    input_ids,mask_idx = encode(bert_tokenizer,text_sentence)
    with torch.no_grad():
        predict= bert_model(input_ids)[0]
    bert = decode(bert_tokenizer,predict[0,mask_idx,:].topk(top_clean).indices.tolist(),top_clean=top_clean)
    return bert

def punctuation_corr(input_sent:str):
    """
    Corrects common anomalies with punctuations such as words starting or
    ending with punctuations and separates them with a white space.

    Parameters
    ----------
    input_sent: str,
        The input document being used for prediction

    Returns
    -------
    str
        The cleaned sentence with punctuations separated by white space
    """
    ## correct punctuation position
    input_split = input_sent.split()
    for i in range(len(input_split)):
        if not input_split[i]: ## for \t\n char 
            continue
        ## word starts with a punctuation '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' but not @ => @abc , may be a tweet mention
        if input_split[i][0] in string.punctuation and input_split[i][0]!='@':
            input_split[i]= input_split[i][0] + " " + input_split[i][1:]

        ## checks for word that ends with puncutation
        if input_split[i][-1] in string.punctuation:
            input_split[i]= input_split[i][:-1] + " " + input_split[i][-1]
            if input_split[i][-3] in string.punctuation: ## accounts for ."
                orig_punc = input_split[i][-3]
                input_split[i]= input_split[i][:-3] +" "+input_split[i][-3:]
                
        ### account for all uppercase words, convert to lower case
        elif input_split[i].upper() == input_split[i] and len(input_split[i])> 1 :  
            input_split[i] = input_split[i].lower()
        
        ## for punct in between the words without space e.g."buy,i"

        try:
            punc_pos = len(re.search('\w+',input_split[i])[0])
            if punc_pos<len(input_split[i])-1:
                if input_split[i][punc_pos] in string.punctuation and input_split[i][punc_pos] !='-' and input_split[i][punc_pos] !="'" :
                    input_split[i] = input_split[i][:punc_pos] + " " +input_split[i][punc_pos]+" "+input_split[i][punc_pos+1:]
        except:
            pass

    input_sent = ' '.join(input_split)
    return input_sent




def find_masked_words(input_sent,top_k=5, useSpacy=True):
    """
    Based on the cleaned text, mask out interested words for BERT prediction and get the BERT prediction
    Two ways to split the sentence, by Spacy or NLTk (Spacy is more advanced and time-consuming)

    Parameters
    ----------
    input_sent: str,
        the input document being used for prediction
    top_k: int,
        the number of prediction get from BERT model for each masked word
    useSpacy : boolean,
        Whether to use Spacy to split words and pos tagging - default is True

    Returns
    -------
    dictionary
        A dictionary of keywords for each document
    """

    keyword = defaultdict(dict)
    if not useSpacy:
        input_sent = punctuation_corr(input_sent)
        input_split = input_sent.split()
            ## second filter with pos tagging for nltk
        ### POS TAGGGING 
        not_consider_pos = ['PRON','DET','ADP','CONJ','NUM','PRT','.',':','CC','CD','EX','DT','PDT',
                        'IN','LS','MD','NNP','NNPS','PRP','POS','PRP$','TO','UH','WDT','WP$','WP','WRB']
            #  refer to   https://www.learntek.org/blog/categorizing-pos-tagging-nltk-python/ 
        be_do_verb = ['is','are','was','were','did','does','do','not','had','have','has','ever']
        conjunction_words = ['therefore','thus']
        pos_res = nltk.pos_tag(input_sent.split())
        for item_num in range(len(pos_res)):
            if pos_res[item_num][1] not in not_consider_pos:
                ## another level of filtering of words before masking 
                if '@' in pos_res[item_num][0]:
                    continue
                if pos_res[item_num][0].lower() in be_do_verb:
                    continue
                if pos_res[item_num][0].lower() in conjunction_words:
                    continue
                if pos_res[item_num][0][-2:] in ["'s","'t","'r","'d","'l","'v","'m"]:
                    continue
                if 'http' in pos_res[item_num][0]:
                    continue
                if len(pos_res[item_num][0])<3:
                    continue
                if pos_res[item_num][0] in stop_words:
                    continue
                if pos_res[item_num][0][-1] in string.punctuation:
                    pos_res[item_num] = pos_res[item_num][:-1]
                orig = input_split[item_num]
                input_split[item_num]='<mask>'
                input_text_for_pred = ' '.join(input_split)
                input_split[item_num]=orig
                keyword[pos_res[item_num][0]+ "_"+str(item_num)]['prediction']=get_predictions(input_text_for_pred, top_clean=top_k)
    else:
        doc = nlp(input_sent)
        input_split = doc.text.split()
        ### reg_exp to detect punctuation and number in the word splits
        reg_exp= "["+string.punctuation+"0-9]"
        for i in range(len(input_split)):
            
            if len(doc[i].text)<3: ## skip words with length < 3
                continue
            if re.search(reg_exp,doc[i].text): ## skip punctuation and number
                continue
            ### remove words that are definitely not emo-denoting for easier computation
            if not doc[i].is_stop and doc[i].pos_ not in ['SPACE','PUNCT','ADX','CONJ','CCONJ',
                                                        'DET','INTJ','NUM','PRON','PROPN','SCONJ','SYM']:
                orig = input_split[i]
                input_split[i]= "<mask>"
                input_text_for_pred = ' '.join(input_split) ### join the split words together with <mask> for BERT prediction
                input_split[i]= orig
                keyword[doc[i].text+ "_"+str(i)]['prediction']=get_predictions(input_text_for_pred, top_clean=top_k)
            
        
    return keyword
    
"""
Auxilary functions to find out the prediction from BERT model, change top_k_choic

"""

### match score pertaining to the masked words with the network metrics/score
## match_col : "Authority" , "modularity_class","Weighted Degree","betweenesscentrality"
def self_score(match_col="Authority",pred_out_pf=None):
    return pred_out_pf['cleaned_index'].map(network[match_col].to_dict())

### match score pertaining to the masked predictions with the network metrics/score
## match_col : "Authority" , "modularity_class","Weighted Degree","betweenesscentrality"
def pred_score(match_col="Authority",pred_out=None):
    pred_score_output = []
    for item in pred_out['prediction']:
        item = item.lower()
        try:
            pred_score_output.append(network[match_col].to_dict()[item])
        except:
            pred_score_output.append(-1)
    return pred_score_output


### aggregate function
def key_word_predict_with_network_from_sent(input_sent=None,top_k=None, filter_NA_pred=True):
    keyword_pred_from_bert_output = find_masked_words(input_sent, top_k=top_k)
    res_out = pd.DataFrame(keyword_pred_from_bert_output).transpose()
    
    if len(res_out) != 0:
        try: 
            res_out['cleaned_index']= [ item.split('_')[0].lower() for item in res_out.index]
            res_out['Label'] = self_score(match_col="Label",pred_out_pf=res_out)## check whether in the network
            res_out['self_auth'] = self_score(match_col="Authority",pred_out_pf=res_out)
            res_out['self_class'] = self_score(match_col="modularity_class",pred_out_pf=res_out)
            res_out['self_deg'] = self_score(match_col="Weighted Degree",pred_out_pf=res_out)
            res_out['self_betcent'] = self_score(match_col="betweenesscentrality",pred_out_pf=res_out)
            res_out['pred_betcent'] = res_out.apply(lambda row: pred_score(match_col="betweenesscentrality",pred_out=row),axis=1)
            res_out['pred_auth'] = res_out.apply(lambda row: pred_score(match_col="Authority",pred_out=row),axis=1)
            res_out['pred_deg'] = res_out.apply(lambda row: pred_score(match_col="Weighted Degree",pred_out=row),axis=1)
            res_out['pred_class'] = res_out.apply(lambda row: pred_score(match_col="modularity_class",pred_out=row),axis=1)
            res_out['string'] = input_sent
            return res_out
        except Exception as e:
            print(str(e))
    else: 
        return None