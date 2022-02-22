"""
Contains methods for straight forward prediction using the trained models,
aswell as creating graphs like confusion matrices

@author: Philipp
"""
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import torch
import torch.nn
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.classification.custom_pytorch_dataset import CustomDataset
from src.classification.nn_models import LSTM_Net


### The dataframe needs an column "embedding" and "label_encoded" e.g. for confusion matrix
###returns previous dataframe incl. now predicted labels and probability
def predict_role_with_true_label(dataframe, model, weight_path):
    dataframe = dataframe.reset_index(drop=True)
    data = CustomDataset(dataframe)
    data_loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=1)

    with torch.no_grad():
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model.load_state_dict(torch.load(weight_path))
        model.to(device)

        model.eval()
        df_predictions = pd.DataFrame(columns=["Index", "True Label", "Predicted Label", "Probability"])
        for sample_batched in data_loader:
            sentences = sample_batched['sentence'].float().to(device)
            labels = sample_batched['label'].float()
            indices = sample_batched['index']
            pred = model(sentences)
            probs = (F.softmax(pred, dim=1)).to('cpu')
            predictions = (torch.argmax(probs, dim=1).numpy())
            for i, idx in enumerate(indices):
                df_predictions = df_predictions.append(
                    {"Index": idx.item(), "True Label": labels.numpy()[i].argmax(), "Predicted Label": predictions[i],
                     "Probability": probs[i].numpy()[predictions[i]]}, ignore_index=True)

        df_predictions = pd.merge(left=dataframe, right=df_predictions,
                                  left_index=True,
                                  right_on="Index", how="outer")
        # maps the label numbers to text
        mapping = pd.DataFrame(columns=['label', 'Predicted Role'])
        mapping = mapping.append({'Predicted Role': 'Citation', 'label': 0}, ignore_index=True)
        mapping = mapping.append({'Predicted Role': 'Evidence', 'label': 1}, ignore_index=True)
        mapping = mapping.append({'Predicted Role': 'Finding', 'label': 2}, ignore_index=True)
        mapping = mapping.append({'Predicted Role': 'Legal Rule', 'label': 3}, ignore_index=True)
        mapping = mapping.append({'Predicted Role': 'Reasoning', 'label': 4}, ignore_index=True)
        mapping = mapping.append({'Predicted Role': 'Sentence', 'label': 5}, ignore_index=True)
        print(df_predictions.info())
        print(mapping.info())
        # merges mapping with predictions
        df_predictions = pd.merge(left=df_predictions, right=mapping, left_on='Predicted Label',right_on="label", how="outer")
        print(df_predictions.info())

        return df_predictions

### needs only a dataframe with 'embedding'
### returns old dataframe now including 'label' the predicted labels and 'prob' = pseudo probabilities
def predict_role(dataframe, model, weight_path):
    dataframe = dataframe.reset_index(drop=True)

    data = CustomDataset(dataframe)
    data_loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=1)

    with torch.no_grad():
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        model.eval()
        df_predictions = pd.DataFrame(columns=["index", "label", "prob"])
        for sample_batched in data_loader:
            sentences = sample_batched['sentence'].float().to(device)
            indices = sample_batched['index']
            pred = model(sentences)
            probs = (F.softmax(pred, dim=1)).to('cpu')
            predictions = (torch.argmax(probs, dim=1).numpy())

            #merges old dataframe with predictions
            for i, idx in enumerate(indices):
                df_predictions = df_predictions.append(
                    {"index": idx.item(), "label": predictions[i], "prob": probs[i].numpy()[predictions[i]]},
                    ignore_index=True)
        df_predictions = pd.merge(left=dataframe, right=df_predictions,
                                  left_index=True,
                                  right_on="index", how="inner")
        #maps the label numbers to text
        mapping = pd.DataFrame(columns=['label', 'role'])
        mapping = mapping.append({'role': 'Citation', 'label': 0}, ignore_index=True)
        mapping = mapping.append({'role': 'Evidence', 'label': 1}, ignore_index=True)
        mapping = mapping.append({'role': 'Finding', 'label': 2}, ignore_index=True)
        mapping = mapping.append({'role': 'Legal Rule', 'label': 3}, ignore_index=True)
        mapping = mapping.append({'role': 'Reasoning', 'label': 4}, ignore_index=True)
        mapping = mapping.append({'role': 'Sentence', 'label': 5}, ignore_index=True)

        #merges mapping with predictions
        df_predictions = pd.merge(left=df_predictions, right=mapping, on="label", how="inner")
        print(df_predictions.head())
        return df_predictions

#print confusion matrix and classification report on given dataset and model
if __name__ == "__main__":
    pd.set_option('display.max_columns', None)

    # weight_path= "../../data/model_weights/balanced_Logistic Regression_DICE_F1_Batch1.dat"
    # weight_path= "../../data/model_weights/balanced_MLP_DICE_F1.dat"
    weight_path = "../../data/model_weights/LSTM Net_balanced_DICE_batch_size_1.dat"
    # weight_path= "../../data/model_weights/Logistic Regression_balanced_DICE_batch_size_1.dat"

    df_sentences = pickle.load(open('../../data/sentences_balanced_legalBERT.p', 'rb'))




    predictions = predict_role_with_true_label(df_sentences[df_sentences.Split == "Test"], LSTM_Net(), weight_path)
    print(predictions.info())
    # predictions.to_csv("../../data/graphs/error_analysis.csv")
    predictions['label_x'] = predictions['label_x'].map(lambda x: x.replace('Sentence','').replace('LegalRule','Legal Rule') if x!='Sentence'else x)
    # predictions = predictions.drop(['label_y'])
    predictions=predictions.rename(columns={'label_x':'Ground Truth Label'})
    print(classification_report(predictions['Ground Truth Label'], predictions['Predicted Role']))
    contingency_matrix = pd.crosstab(predictions['Ground Truth Label'], predictions['Predicted Role'])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = sn.heatmap(contingency_matrix.T, annot=True, fmt='.2f', cmap="YlGnBu", cbar=False)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("../../data/graphs/confusion_matrix_lstm_test.png")
    plt.show()
