"""
Created on Sun Dec 05 18:28:59 2021

@author: Philipp , Sarah , Johannes
"""

import json
import pandas as pd
import pickle

from enum import Enum
from pathlib import Path
from pydantic import BaseModel
# from sklearn.model_selection import train_test_split
import sentence_encoder

class SentLabel(Enum):
    FINDING_SENTENCE = "FindingSentence"
    REASONING_SENTENCE = "ReasoningSentence"
    EVIDENCE_SENTENCE = "EvidenceSentence"
    LEGAL_RULE_SENTENCE = "LegalRuleSentence"
    CITATION_SENTENCE = "CitationSentence"
    SENTENCE = "Sentence"

class Sentence(BaseModel):
    sent_id: str
    doc_id: str
    dataset_type: int
    sent_label: SentLabel
    sent: str

class Document(BaseModel):
    doc_id: str
    dataset_type: int
    doc_text: str




'''Created by Sarah Lockfisch'''
class DataProvider:
    
    def __init__(self, sentences_folder_path):
        self.sentences_folder_path = sentences_folder_path
        self._json_files = list(Path(sentences_folder_path).glob("**/*.json"))
        self._sentences_dict = dict()
        self._documents_dict = dict()


    @property
    def sentences(self) -> Sentence:
        if len(self._sentences_dict) == 0:
            self._init_from_files()
        return self._sentences_dict
        
    @property
    def documents(self) -> Document:
        if len(self._documents_dict) == 0:
            self._init_from_files()
        return self._documents_dict

    def _read_json_files(self, json_files):
        for i in range(len(json_files)):
                with open(json_files[i], 'r',encoding="utf8") as file:
                    yield json.load(file)

    def _init_from_files(self):
        for i,file_json in enumerate(self._read_json_files(self._json_files)):
            if "docID" in file_json:
                document = Document(
                    doc_id=file_json["docID"],
                    dataset_type=1,
                    doc_text=file_json["text"]
                    )
                self._documents_dict[document.doc_id] = document
                for sentence_data in file_json["sentences"]:
                    sentence = Sentence(
                        sent_id=sentence_data["sentID"],
                        doc_id=file_json["docID"],
                        dataset_type =1,
                        sent_label=SentLabel(sentence_data["rhetRole"][0]),
                        sent=sentence_data["text"]
                    )
                    self._sentences_dict[sentence.sent_id] = sentence
            else:
                print(file_json['caseNumber'])
                document = Document(
                    doc_id=file_json["caseNumber"],
                    dataset_type =0,
                    doc_text=file_json["text"]
                )
                self._documents_dict[document.doc_id] = document
                for sentence_data in file_json["sentences"]:
                    sentence = Sentence(
                        sent_id=sentence_data["sentID"],
                        doc_id=file_json["caseNumber"],
                        dataset_type=0,
                        sent_label=SentLabel(sentence_data["rhetClass"]),
                        sent=sentence_data["text"]
                    )
                    self._sentences_dict[sentence.sent_id] = sentence



    def get_dataframes(self):
        rows_doc = []
        rows_sent = []
        for file_json in self._read_json_files(self._json_files):
            if "docID" in file_json:
                rows_doc.append([file_json['docID'],1,'Train', file_json['text']])
                for sentence_data in file_json["sentences"]:
                    rows_sent.append([sentence_data["sentID"],file_json['docID'],1,  sentence_data['rhetRole'][0],'Train',
                                      sentence_data['text']])
            else:
                rows_doc.append([file_json['caseNumber'],0,'Train', file_json['text']])
                for sentence_data in file_json["sentences"]:
                    rows_sent.append([sentence_data["sentID"],
                                      file_json['caseNumber'],
                                      0,
                                      sentence_data['rhetClass'][0],'Train', sentence_data['text']])

        df_documents = pd.DataFrame(rows_doc, columns=['doc_id','dataset_type','Split', 'text'])
        df_sentences = pd.DataFrame(rows_sent, columns=['sent_id','doc_id','dataset_type',  'label','Split', 'text'])
        return df_documents, df_sentences

    '''Created by Johannes, modified Philipp'''
    #start & end position is counted in characters
def get_sentence_position(df_documents,df_sentences):
    # old columns of df_sentences ['sent_id', 'doc_id', 'label', 'text']


    df_sentences['start_pos'] = 0
    df_sentences['end_pos'] = 0

    for index, docs in df_documents.iterrows():
        doc_id = docs['doc_id']
        doc_text = docs['text']
        df_sentences[df_sentences.doc_id == doc_id]
        for index, sent in df_sentences.iterrows():
            if doc_id == sent['doc_id']:
                sentence = sent['text']
                start_pos = doc_text.find(sentence)
                if start_pos != -1:
                    df_sentences.loc[(df_sentences.sent_id == sent['sent_id']), 'start_pos'] = start_pos
                    df_sentences.loc[(df_sentences.sent_id == sent['sent_id']), 'end_pos'] = start_pos+len(sentence)




    #new columns in new_df_sentences ['sent_id', 'doc_id', 'label', 'text','start_pos','end_pos']
    df_sentences['start_pos'] = 0
    df_sentences['end_pos'] = 0

    for index, docs in df_documents.iterrows():
        doc_id = docs['doc_id']
        doc_text = docs['text']
        df_sentences[df_sentences.doc_id == doc_id]
        for index, sent in df_sentences.iterrows():
            if doc_id == sent['doc_id']:
                sentence = sent['text']
                start_pos = doc_text.find(sentence)
                if start_pos != -1:
                    df_sentences.loc[(df_sentences.sent_id == sent['sent_id']), 'start_pos'] = start_pos
                    df_sentences.loc[(df_sentences.sent_id == sent['sent_id']), 'end_pos'] = start_pos + len(sentence)

    # new columns in new_df_sentences ['sent_id', 'doc_id', 'label', 'text','start_pos','end_pos']
    new_df_sentences = df_sentences
    return new_df_sentences

'''Created by Philipp'''
if __name__ == "__main__":
    #This parses the Json Files and creates two pandas DataFrames that yet not entail the sentence position (characterwise)
    sentences_folder_path = Path("../data")
    provider = DataProvider(sentences_folder_path)
    df_documents, df_sentences = provider.get_dataframes()

    #This enriches the df_sentences DataFrame with the sentence position
    df_sentences=get_sentence_position(df_documents,df_sentences)

    #Encode Sentences with legal bert
    df_sentences = sentence_encoder.sentence_bert_embeddings(df_sentences,model=1)

    #Encode Sentence Labels
    df_sentences.loc[df_sentences['label'] == 'S', 'label'] = "Sentence"
    df_sentences.loc[df_sentences['label'] == 'R', 'label'] = "ReasoningSentence"
    df_sentences.loc[df_sentences['label'] == 'E', 'label'] = "EvidenceSentence"
    df_sentences.loc[df_sentences['label'] == 'C', 'label'] = "CitationSentence"
    df_sentences.loc[df_sentences['label'] == 'F', 'label'] = "FindingSentence"
    df_sentences.loc[df_sentences['label'] == 'L', 'label'] = "LegalRuleSentence"

    label_encoding={'CitationSentence': 0, 'EvidenceSentence': 1, 'FindingSentence': 2, 'LegalRuleSentence': 3, 'ReasoningSentence': 4, 'Sentence': 5}
    df_sentences["label_encoded"] = df_sentences["label"].map(label_encoding)



    #While the code above does work, it is better to hardcode the documents so we can be sure that NOONE touches the Test set
    test_documents =['1400029','1431031','1316146','1456911','1525217','1554165','1607479','1710389','18161103','19139412','19154420']# 8 documents dataset Type 1 + 3 documents dataset Type 0
    validation_documents =['1705557','1713615','1315144','1340434','1715225','1719263','1731026','19160065','19161706','18139471']# 7 documents dataset Type 1 + 3 documents dataset Type 0

    df_documents.loc[df_documents.doc_id.isin(test_documents), "Split"] = "Test"
    df_documents.loc[df_documents.doc_id.isin(validation_documents), "Split"] = "Validation"
    df_sentences.loc[df_sentences.doc_id.isin(test_documents), "Split"] = "Test"
    df_sentences.loc[df_sentences.doc_id.isin(validation_documents), "Split"] = "Validation"

    # Create Balanced
    g = df_sentences[df_sentences.Split == "Train"].groupby("label")
    train_set_balanced = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    g = df_sentences[df_sentences.Split == "Validation"].groupby("label")
    val_set_balanced = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    g = df_sentences[df_sentences.Split == "Test"].groupby("label")
    test_set_balanced = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))

    frames_for_balanced = [train_set_balanced, val_set_balanced,test_set_balanced]
    df_balanced = pd.concat(frames_for_balanced)

    #consists of all training sentences, but same balanced validation and test set as in balanced dataset
    frames_for_full = [df_sentences[df_sentences.Split == "Train"], val_set_balanced,test_set_balanced]
    df_full = pd.concat(frames_for_full)


    print(df_documents.info())
    print(df_documents[df_documents.Split == "Test"].info())
    print(df_documents[df_documents.Split == "Validation"].info())
    print(df_documents[df_documents.Split == "Train"].info())

    print(df_full.info())
    print(df_full[df_full.Split == "Test"].info())
    print(df_full[df_full.Split == "Validation"].info())
    print(df_full[df_full.Split == "Train"].info())

    print(df_balanced.info())
    print(df_balanced[df_balanced.Split == "Test"].info())
    print(df_balanced[df_balanced.Split == "Validation"].info())
    print(df_balanced[df_balanced.Split == "Train"].info())

    #Store documents & sentences in a  efficient file format
    pickle.dump(df_documents, open("../data/documents.p", "wb"))
    pickle.dump(df_full, open("../data/sentences_full_legalBERT.p", "wb"))
    pickle.dump(df_balanced, open("../data/sentences_balanced_legalBERT.p", "wb"))


