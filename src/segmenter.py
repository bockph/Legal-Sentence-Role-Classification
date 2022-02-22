"""
This File contains everything that is needed for adapting the spacy sentence segmenter
@author: Surim/David,Johannes, Prodyumna, Philipp
"""

import re

import pandas as pd
from spacy.language import Language
from spacy.symbols import ORTH
from spacy.tokens import Token
from tqdm import tqdm

### Evaluation with ablation study
## baseline, when nothing ablated
# TP: 1831 FP: 679 FN: 420
# F1: 0.7691661415689005 Precision: 0.7294820717141474 Recall: 0.8134162594412487

####
# Some Informaiton on Tokens https://spacy.io/api/token

##
# https://spacy.io/usage/linguistic-features#special-cases
# This basically means find terms that have one connected meaning but are lexicographically  multiple words signs etc.
# Why? Because otherwise spacy might split them into several tokens
##
special_cases = [
    # from Johannes
    ('Vet. App.', [{ORTH: 'Vet. App.'}]),
    ('Fed. Reg.', [{ORTH: 'Fed. Reg.'}]),
    ('CONCLUSION OF LAW', [{ORTH: 'CONCLUSION OF LAW'}]),
    ('CONCLUSIONS OF LAW', [{ORTH: 'CONCLUSIONS OF LAW'}]),
    ('Id.', [{ORTH: 'Id.'}]),
    # from Prodyumna
    ('FINDINGS OF FACT', [{ORTH: 'FINDINGS OF FACT'}]),
    ('REASONS AND BASES FOR FINDINGS AND CONCLUSION', [{ORTH: 'REASONS AND BASES FOR FINDINGS AND CONCLUSION'}]),
    ('ATTORNEY FOR THE BOARD', [{ORTH: 'ATTORNEY FOR THE BOARD'}]),
    ('THE ISSUE', [{ORTH: 'THE ISSUE'}])
]

##

# https://spacy.io/usage/rule-based-matching#matcher
# TODO: Define  Token attributes here.
#
##

attribute_ruler_patterns = [
    # do not delete for evaluation / ablation study because necessary for Johannes' patterns later on
    # from Johannes (as David believes)
    # pattern to identify Headings and Linebreaks after Heading and assign them the Heading Attribute
    # see also https://explosion.ai/demos/matcher
    {"patterns": [[{'IS_UPPER': True, 'OP': '+'}, {'IS_SPACE': True}]], "attrs": {"_": {'HEADING': True}}},
]


def set_token_extensions():
    Token.set_extension('CASEREFERENCE', default=False, force=True)
    Token.set_extension('UNIMPORTANT', default=False, force=True)
    Token.set_extension('HEADING', default=False, force=True)
    Token.set_extension('PARAGRAPHS', default=False, force=True)
    # from Johannes (as David believes)
    Token.set_extension('PARAGRAPH', default=False, force=True)


## Spacy component to improve sentenicer. Iterates over all found tokens, and decided wether they are a sentence start or not
@Language.component("improved_sentenicer")
def set_custom_Sentence_end_points(doc):
    set_token_extensions()
    inbracket = False
    for index, token in enumerate(doc[:-1]):

        ## from Philip
        ###Example for how to improve the segmenter, do not forget to comment this when evaluating the baseline
        ###heading_beginning
        if token.text.isupper() and doc[token.i + 1].text.isupper():
            if index != 0:  # If not first word in document
                if not doc[token.i - 1].text.isupper():
                    token.is_sent_start = True  # David: different when deleting "= True" (like this line was before)
                else:
                    token.is_sent_start = False

        ###Handle Trailing Whitespaces for other BVA documents
        if re.search(r'[0-9]', token.text) and re.search(r'(\r\n\r\n)', doc[token.i - 1].text):
            if token.is_sent_start == None:
                token.is_sent_start = True
        if re.search(r'(\r\n){2,}|(\t\r\n)|(\n\n)|(\n\t)', token.text):
            if token.is_sent_start == None:
                token.is_sent_start = True
            if doc[token.i + 1].is_sent_start == None:
                doc[token.i + 1].is_sent_start = True

        ###Handle Newlines after commas and dots
        if re.search(r'(\r\n){1}', token.text) and re.search(r',', doc[token.i - 1].text):
            # print(repr(token.text))
            if token.is_sent_start == None:
                token.is_sent_start = False
            if doc[token.i + 1].is_sent_start == None:
                doc[token.i + 1].is_sent_start = False

        if re.search(r'(\r\n){1}', token.text) and re.search(r'\.', doc[token.i - 1].text) and len(
                doc[token.i - 2].text) > 3:
            if doc[token.i + 1].is_sent_start == None:
                doc[token.i + 1].is_sent_start = True

        if token.text == 'Archive' and doc[token.i + 1].text == 'Date':
            token.is_sent_start = True

        ## from David
        # num_or_lowcase_after_heading
        # Word after a heading should be a stentence-start
        # current token: Number | (Word with) first letter is uppercase
        # token before: linebreak
        if (token.like_num | token.text[0].isupper()):
            if ((doc[token.i - 1].text == "\r\n") | (doc[token.i - 1].text == "\r\n\r\n") | (
                    doc[token.i - 1].text == "\n")):
                token.is_sent_start = True

        ## from Johannes
        # Another example to treat Headings,by using the special cases above they may be hardcoded
        # if token._.HEADING: print(token.text)

        # token_after_heading
        # tokens that are not classified as headings but have a classified heading token before them, those are a new sentence start
        if doc[token.i - 1]._.HEADING == True and token._.HEADING == False:
            token.is_sent_start = True

        # curly_brackets
        # whenever something is inside curly brackets, it's never supposed to start a new sentence
        if token.text == '(':
            inbracket = True
        if token.text == ')':
            inbracket = False
        if inbracket:
            token.is_sent_start = False

        # hard_code_id. As a stand-alone sentence
        if token.text == 'Id.':
            token.is_sent_start = True
            doc[token.i + 1].is_sent_start = True

        # identify_whitespaces & linebreaks (which are also whitespace tokens in spacy) -> set next token as sentence_start (F 0.6 -> 0.8)
        if token.is_space == True:
            doc[token.i + 1].is_sent_start = True

        ## from Prodyumna
        # Both below IFs give result: TP: 1340 FP: 788 FN: 911
        # break_after_uppercase
        if token.text[0].isupper() and doc[token.i + 1].text == "\r\n":
            doc[token.i + 1].is_sent_start = True

        # num_punct_neighbours
        if (token.is_punct and doc[token.i - 1].like_num) | (token.like_num and doc[token.i - 1].is_punct):
            token.is_sent_start = False

        # Below IF with above two IFs gives result: TP: 1400 FP: 818 FN: 851 | F1: 0.6265
        # identify_linebreaks
        if token.text == "\r\n\r\n":
            doc[token.i + 1].is_sent_start = True

    return doc


def tokenizer(spacy_sentences, doc_id, split):
    final_sentences = []
    for sentence in spacy_sentences:
        if len(sentence.text.strip()) <= 1:
            pass
        else:
            tokens = []
            # TODO implement Tokens for token based embeddings

            sentence_prediction = [doc_id, sentence.start_char, sentence.end_char, split, sentence.text, tokens]
            final_sentences.append(sentence_prediction)

    return pd.DataFrame(final_sentences, columns=["doc_id", "start_pos", "end_pos", "Split", "text", "tokens"])


# Function to match ground truth sentences and predicted sentences
# uses the start and end char positions of a sentence and compares against a set threshold
# returns positive/negative rates
def evaluate_segmenter(ground_truth_sentences, generated_sentences, threshold=3):
    # |Starting Char GT Sentence - Starting Char Generated Sentence| +|Ending Char GT Sentence - Ending Char Generated Sentence|<threshold

    ground_truth_sentences['matched'] = False
    ground_truth_sentences = ground_truth_sentences.sort_values(by=['start_pos'])

    generated_sentences['matched'] = False
    generated_sentences = generated_sentences.sort_values(by=['doc_id', 'start_pos'])

    TP = 0  # matched true and generated splits
    generated_sentences = generated_sentences.reset_index(drop=True)
    for index_gt, rows_gt in tqdm(ground_truth_sentences.iterrows()):
        doc_id = rows_gt['doc_id']
        if rows_gt['matched']:
            continue
        for index_pd, rows_pd in generated_sentences[generated_sentences.doc_id == doc_id].iterrows():
            if (abs(rows_gt['start_pos'] - rows_pd['start_pos']) + abs(
                    rows_gt['end_pos'] - rows_pd['end_pos'])) <= threshold:  # Philip
                # if (abs(rows_gt['start_pos'] - rows_pd['start_pos'])) <= threshold:  # David: results in F1: 0.699 and TP: 1616 FP: 755 FN: 635
                ground_truth_sentences.at[index_gt, "matched"] = True
                generated_sentences.at[index_pd, "matched"] = True
                TP += 1
                break
    # David: F means "did not match"
    FN = len(ground_truth_sentences[
                 ground_truth_sentences.matched == False])  # Count Unmatched Entries in ground Truth sentences # David: It's a sentence in the ground truth df but we did not predict it to be a sentence (in the generated df)
    FP = len(generated_sentences[
                 generated_sentences.matched == False])  # Count Unmatched Entries in generated sentences # David: We predicted it to be a sentence (in generated df) but this sentence did not show up in the ground truth df
    return (TP, FP, FN, ground_truth_sentences, generated_sentences)
