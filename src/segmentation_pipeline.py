"""
Provides the framework for sentence segmentation

@author: Philipp
"""

import pickle

import pandas as pd
import spacy

from src import segmenter as segmenter


# takes a document string and parses it into sentences using the segmenter.py model
# returns a pandas dataframe with ['text','start_char'] of each sentence
def run_segmenter(document):
    segmenter.set_token_extensions()
    nlp = spacy.load("en_core_web_lg")

    for special_case in segmenter.special_cases:
        nlp.tokenizer.add_special_case(*special_case)
    nlp.add_pipe('improved_sentenicer', before='parser')
    nlp.remove_pipe("attribute_ruler")
    ruler = nlp.add_pipe("attribute_ruler", before="improved_sentenicer")
    nlp.remove_pipe("lemmatizer")
    nlp.remove_pipe("ner")
    ruler.add_patterns(segmenter.attribute_ruler_patterns)

    generated_sentences_list = nlp(document).sents
    df_sentences = pd.DataFrame(columns=["text"])

    for sentence in generated_sentences_list:
        if "".join(sentence.text.split()) != "":
            df_sentences = df_sentences.append({'text': sentence.text, 'start_char': sentence.start_char},
                                               ignore_index=True)
    df_sentences.dropna(inplace=True)

    return df_sentences


###Running Main one can evaluate the current segmentation model of segmenter.py
if __name__ == "__main__":
    # Initiate Spacy Model
    segmenter.set_token_extensions()
    nlp = spacy.load("en_core_web_md")

    # Load Dataset
    df_documents = pickle.load(open("../data/documents.p", 'rb'))
    df_sentences = pickle.load(open('../data/sentences_full_legalBERT.p.p', 'rb'))

    for special_case in segmenter.special_cases:
        nlp.tokenizer.add_special_case(*special_case)
    nlp.add_pipe('improved_sentenicer', before='parser')
    ruler_bytes = nlp.get_pipe("attribute_ruler").to_bytes()
    nlp.remove_pipe("attribute_ruler")
    ruler = nlp.add_pipe("attribute_ruler", before="improved_sentenicer")
    nlp.remove_pipe("lemmatizer")
    nlp.remove_pipe("ner")
    # ruler.from_bytes(ruler_bytes)
    ruler.add_patterns(segmenter.attribute_ruler_patterns)
    generated_sentences_list = []

    # select data to work with
    ## Split
    df_documents = df_documents[df_documents.Split.str.match("Train") | df_documents.Split.str.match('Validation')]
    df_sentences = df_sentences[df_sentences.Split.str.match("Train") | df_sentences.Split.str.match('Validation')]

    ## dataset_type --> We use the documents that allow segmentation testing (Type 0)
    df_documents = df_documents[df_documents.dataset_type == 0]
    df_sentences = df_sentences[df_sentences.dataset_type == 0]

    for index, row in df_documents.iterrows():
        # Run Spacy pipeline to get all sentences
        generated_sentences_raw = nlp(row['text']).sents

        # This enriches the sentences with a segmentation into Tokens and returns the dataframe
        generated_sentences_list.append(
            segmenter.tokenizer(generated_sentences_raw, doc_id=row['doc_id'], split=row['Split']))

    df_generated_sentences = pd.concat(generated_sentences_list)
    print(df_generated_sentences.info())
    print(df_sentences.info())

    # evaluate model
    TP, FP, FN, df_ground_truth_sentences_with_matches, df_generated_sentences_with_matches = segmenter.evaluate_segmenter(
        ground_truth_sentences=df_sentences, generated_sentences=df_generated_sentences, threshold=3)

    #  Qualitative Analysis:
    df_ground_truth_sentences_with_matches.to_csv("../data/output_ground_truth_sentences.csv")
    df_generated_sentences_with_matches.to_csv("../data/output_generated_sentences.csv")

    # Quantitative Analysis: calculate F1, Precision and Recall
    precision = TP / (TP + FP) + 0.000000000001
    recall = TP / (TP + FN) + 0.000000000001
    F1 = (2 * precision * recall) / (precision + recall) + 0.000000000001
    print('TP: {} FP: {} FN: {}'.format(TP, FP, FN))
    print('F1: {} Precision: {} Recall: {}'.format(F1, precision, recall))
