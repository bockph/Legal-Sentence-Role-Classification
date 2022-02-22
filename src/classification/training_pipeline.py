"""
This File contains everything needed for the training of the neural networks
@author: Philipp
"""

import pickle
import traceback
from time import time

import pandas as pd
import torch
import torch.nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_loss_functions import DiceLoss, AUC
from custom_pytorch_dataset import CustomDataset
from nn_models import Logistic_Regression,LSTM_Net,ConvolutionalNet


def validate_model(model, criterion, data_loader):
    # Fixed criterion used for comparing different loss functions
    try:
        general_criterion = DiceLoss()

        with torch.no_grad():
            device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
            model.to(device)
            model.eval()

            running_loss = 0
            running_dice = 0
            running_auc_roc = 0
            running_auc_pr = 0
            for sample_batched in tqdm(data_loader):

                sentences = sample_batched['sentence'].float().to(device)
                labels = sample_batched['label'].float().to(device)

                pred = model(sentences)
                loss = criterion(pred, labels)

                dice = general_criterion(pred, labels)

                auc_roc, auc_pr, points = AUC(pred.to('cpu'), labels.to('cpu'))

                running_loss += loss.item()
                running_dice += dice.item()
                running_auc_pr += auc_pr
                running_auc_roc += auc_roc

            else:
                return [running_loss / len(data_loader), running_dice / len(data_loader),
                        running_auc_roc / len(data_loader), running_auc_pr / len(data_loader)]
    except Exception as e:
        print("Validation failed with: " + str(e))
        traceback.print_exc()
        return [-1, -1]


def train_model(model, criterion, criterion_name, optimizer, dataset_name, batch_size, train_loader, validation_loader,
                num_epochs,
                model_weights_path, model_loss_data_path, pretrained_model=None):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    # Load weights of pretrained model
    if pretrained_model:
        model.load_state_dict(torch.load(pretrained_model))
    min_validation_loss = -1
    best_epoch = 0
    # scaler is needed for Mixed Precision Training
    # scaler = torch.cuda.amp.GradScaler()
    data = pd.DataFrame(
        columns=['Epoch', 'TrainingLoss', 'ValidationLoss', 'ValidationDiceLoss', 'AUC_ROC', 'AUC_PR'])
    for e in range(num_epochs):
        try:
            time0 = time()
            model.train()
            running_loss = 0
            print("Epoch: " + str(e))
            for sample_batched in tqdm(train_loader):
                optimizer.zero_grad()

                # images, masks in trainloader:
                sentences = sample_batched['sentence'].to(device)
                labels = (sample_batched['label']).to(device=device)

                # Autocast is needed for Mixed-Precision Training
                # with torch.cuda.amp.autocast():
                #     predicted_masks = model(images)
                #     loss = criterion(predicted_masks.squeeze(), true_masks)

                # scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()

                pred = model(sentences)
                loss = criterion(pred, labels)

                running_loss += loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            else:

                training_loss = running_loss / len(train_loader)
                validation_loss = validate_model(model, criterion, validation_loader)
                if e == 0 or (min_validation_loss > validation_loss[0]):
                    min_validation_loss = validation_loss[0]
                    best_epoch = e
                    torch.save(model.state_dict(),
                               model_weights_path + model.name + '_' + dataset_name + '_' + criterion_name + '_batch_size_' + str(
                                   batch_size) + ".dat")
                data = data.append({'Epoch': e, 'TrainingLoss': training_loss, 'ValidationLoss': validation_loss[0],
                                    'ValidationDiceLoss': validation_loss[1], 'AUC_ROC': validation_loss[2],
                                    'AUC_PR': validation_loss[3]}, ignore_index=True)
                data.to_csv(
                    model_loss_data_path + model.name + '_' + dataset_name + '_' + criterion_name + '_batch_size_' + str(
                        batch_size) + '_LossData.csv', )

                print("Epoch {} - Training loss: {} - Validation loss: {} - Dice Validation Loss: {}  "
                      .format(e, training_loss, validation_loss[0], validation_loss[1]))
                print("AUC ROC: {} - AUC PR: {}".format(validation_loss[2], validation_loss[3]))

        except Exception as ex:
            print("Training Epoch: " + str(e) + ' failed with: ' + str(ex))
            traceback.print_exc()

    data = data.append({'Epoch': best_epoch, 'TrainingLoss': 0, 'ValidationLoss': min_validation_loss,
                        'ValidationDiceLoss': 0, 'AUC_ROC': 0,
                        'AUC_PR': 0,
                        'TrainingTime': 0}, ignore_index=True)
    data.to_csv(model_loss_data_path + model.name + '_' + dataset_name + '_' + criterion_name + '_batch_size_' + str(
        batch_size) + '_LossData.csv', )


def training_with_grid_search(datasets, models):
    ###
    # Select Max Number of Epochs
    ###
    num_epochs = 40

    ####
    # Select the different Loss functions to run
    ####
    criterions = [(torch.nn.CrossEntropyLoss(), 'CE'), (DiceLoss(), 'DICE')]
    # ,
    ###
    # Select the different Batchsizes to run
    ###
    batch_sizes = [1]

    # Specify Learning Rate
    learning_rate = 0.001
    # Number Workers for data loader
    for model in models:
        for dataset in datasets:
            for batch_size in batch_sizes:
                train_data = dataset['data'][dataset['data'].Split == "Train"].reset_index(drop=True)
                val_data = dataset['data'][dataset['data'].Split == "Validation"].reset_index(drop=True)

                train_set = CustomDataset(train_data)
                val_set = CustomDataset(val_data)
                for criterion in criterions:

                    try:
                        print(
                            "Now training: Model " + model.name + " LossFunction " + criterion[1] + "Batchsize " + str(
                                batch_size))
                        # dataloaders to specify batch size
                        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=1)
                        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=1)

                        # current Model is also the name used for storing weights, and loss data
                        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                        train_model(
                            model=model, criterion=criterion[0], criterion_name=criterion[1], optimizer=optimizer,
                            num_epochs=num_epochs, dataset_name=dataset['name'], batch_size=batch_size,
                            train_loader=train_loader, validation_loader=val_loader,
                            model_weights_path="../data/model_weights/",
                            model_loss_data_path="../data/loss_data/"
                        )

                        # Might help with some Issues on the clinic DL Server
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print("Procedure failed with: " + str(e))


if __name__ == "__main__":
    # Load Dataset
    df_sentences_full = pickle.load(open('../data/sentences_full_legalBERT.p', 'rb'))
    df_sentences_balanced = pickle.load(open('../data/sentences_balanced_legalBert.p', 'rb'))
    datasets = [{"data": df_sentences_balanced, "name": "balanced"}, {"data": df_sentences_full, "name": "full"}]

    # This starts training + Validation procedure LSTM_Net(),ConvolutionalNet()]),
    training_with_grid_search(datasets, [Logistic_Regression(),LSTM_Net(),ConvolutionalNet()])
