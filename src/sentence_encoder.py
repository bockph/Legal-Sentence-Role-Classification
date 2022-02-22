"""
This File contains methods to encode text into embeddings
@author: Philipp
"""
import numpy as np
import tensorflow_hub as hub
# Imports sentence bert stuf
from sentence_transformers import SentenceTransformer


#TODO this is not tested yet
def usc_embeddings(df_sentences):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print("module %s loaded" % module_url)
    for index, row in df_sentences.iterrows():
        df_sentences[index, 'embedding'] = model([row['text']])[0]
    return df_sentences

#sentence bert
def sentence_bert_embeddings(df_sentences,model=0):
    df_sentences['embedding']=np.nan
    df_sentences['embedding']=df_sentences['embedding'].astype(object)
    sbert_model=0

    if model == 1:
        sbert_model = SentenceTransformer('nlpaueb/legal-bert-base-uncased')
    else:
        sbert_model = SentenceTransformer('all-mpnet-base-v2')


    for index,row in df_sentences.iterrows():
        sentence =row['text']
        embedding =sbert_model.encode(sentence).tolist()
        df_sentences.at[index,'embedding']=embedding

    return df_sentences
