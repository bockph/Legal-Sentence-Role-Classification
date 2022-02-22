"""
Creates a Uvicorn webapp backend
@author: Philipp
"""

import os

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import src.classification.prediction as classifier
import src.sentence_encoder as sentence_encoder
from src import segmentation_pipeline as segmentation_pipeline
from src.classification.nn_models import LSTM_Net


class Document(BaseModel):
    text: str


app = FastAPI()

templates = Jinja2Templates(directory="web_app_templates")


@app.post("/doc")
async def perform_eval(document: Document):
    # document contains the text submitted via the webpage
    # 1. segment into sentences
    sentences = segmentation_pipeline.run_segmenter(document.text)
    # 2. encode with bert
    encoded_sentences = sentence_encoder.sentence_bert_embeddings(sentences, model=1)
    # 3 get model and do prediction
    path = os.path.dirname(os.path.abspath(__file__))
    weight_path = path + "\..\data\model_weights\LSTM Net_balanced_DICE_batch_size_1.dat"
    predicted_sentences = classifier.predict_role(encoded_sentences, LSTM_Net(), weight_path)
    # 4. sort sentences by occuring in document
    predicted_sentences = predicted_sentences.sort_values('start_char', ascending=True)

    response = []
    # round probabilities
    predicted_sentences['prob'] = predicted_sentences['prob'].round(2)

    # create dataobject for webpage
    for index, sentence in predicted_sentences.iterrows():
        response.append({"sentence": sentence['text'], "role": sentence['role'], "prob": sentence['prob']})

    return response


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/styles.css")
async def css(request: Request):
    return templates.TemplateResponse("style.css", {"request": request})
