"""
Creates some graphs regarding dataset

@author: Philipp
"""

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_dataset():
    df_sentences = pickle.load(open('../data/sentences_full_legalBERT.p', 'rb')).reset_index(drop=True)
    pd.set_option('display.max_columns', None)
    print(df_sentences.info())

    g = df_sentences[df_sentences.Split == "Validation"].groupby("label")
    val_set_balanced = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
    g = df_sentences[df_sentences.Split == "Test"].groupby("label")
    test_set_balanced = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))

    frames_for_balanced = [df_sentences[df_sentences.Split == "Train"], val_set_balanced, test_set_balanced]
    df_sentences = pd.concat(frames_for_balanced)

    sentence = df_sentences[df_sentences.label == "Sentence"]
    citation = df_sentences[df_sentences.label == "CitationSentence"]
    evidence = df_sentences[df_sentences.label == "EvidenceSentence"]
    legalrule = df_sentences[df_sentences.label == "LegalRuleSentence"]
    reasoning = df_sentences[df_sentences.label == "ReasoningSentence"]
    finding = df_sentences[df_sentences.label == "FindingSentence"]

    labels = ['Custom Sentence', 'Finding', 'Reasoning', 'Legal Rule', 'Evidence', 'Citation']

    type_train = [sentence[sentence.Split == "Train"].shape[0], finding[finding.Split == "Train"].shape[0],
                  reasoning[reasoning.Split == "Train"].shape[0],
                  legalrule[legalrule.Split == "Train"].shape[0], evidence[evidence.Split == "Train"].shape[0],
                  citation[citation.Split == "Train"].shape[0]]
    type_val = [sentence[sentence.Split == "Validation"].shape[0], finding[finding.Split == "Validation"].shape[0],
                reasoning[reasoning.Split == "Validation"].shape[0],
                legalrule[legalrule.Split == "Validation"].shape[0], evidence[evidence.Split == "Validation"].shape[0],
                citation[citation.Split == "Validation"].shape[0]]
    type_test = [sentence[sentence.Split == "Test"].shape[0], finding[finding.Split == "Test"].shape[0],
                 reasoning[reasoning.Split == "Test"].shape[0],
                 legalrule[legalrule.Split == "Test"].shape[0], evidence[evidence.Split == "Test"].shape[0],
                 citation[citation.Split == "Test"].shape[0]]

    width = 0.35
    fig, ax = plt.subplots()
    fig.set_size_inches(9, 5)

    ax.bar(labels, type_train, width, label='Train')
    ax.bar(labels, type_val, width, label='Validation', bottom=type_train)
    ax.bar(labels, type_val, width, label='Test', bottom=np.array(type_train) + np.array(type_val))

    ax.set_ylabel('Number Sentences')
    ax.set_title('Rhetorical Roles')
    ax.legend()
    plt.tight_layout()
    fig.savefig('../data/graphs/distribution_rhetorical_roles_full.png', dpi=100)
    plt.show()


def plot_batch_sizes_across_models():
    # plt.figure(figsize=(8,7))
    fig, ((ax1, ax2, ax3)) = plt.subplots(1, 3, figsize=(10, 3), sharex=True, sharey=True,
                                          subplot_kw={'ylim': 0, }, gridspec_kw={'hspace': .2, 'wspace': 0.1}
                                          )
    ax = (ax1, ax2, ax3)
    print(ax)
    # fig.suptitle('Comparison of Ba' + model + ' and Dice Loss')
    lines = []
    models = ['LSTM Net', 'Logistic Regression', 'Convolutional Net']
    batch_sizes = [1, 32, 128]
    model_loss_data_path = "../data/loss_data/"
    for i, model in enumerate(models):

        for batch_size in batch_sizes:

            label = batch_size  # +'_'+loss_function
            try:
                data = pd.read_csv(
                    model_loss_data_path + model + '_' + 'balanced' + '_' + 'DICE' + '_batch_size_' + str(
                        batch_size) + '_LossData.csv')[:40]
                ax[i].set_title(model)
                print(data['ValidationLoss'].min())
                # ax[i].axhline(y=data['ValidationLoss'].min(), color='r', linestyle='-')
                # if dataset == "All_Annotations_Normalized":
                #     ax[i].set_title("Unbalanced")

                line = ax[i].plot(
                    data.Epoch,
                    data.ValidationDiceLoss,
                    label=label
                )

                if i == 0:
                    lines.append(line)
                ax[i].grid()
            except Exception as e:
                print("Failed: " + str(e))
    plt.subplots_adjust(top=0.9, bottom=0.17)
    fig.tight_layout(pad=2)
    fig.text(.5, .04, 'Epoch', ha='center')
    # fig.text(0.04, 0.5, 'Validation Dice Loss (= 1 - F1 Score)', va='center', rotation='vertical')
    fig.text(0.04, 0.5, 'Dice Loss', va='center', rotation='vertical')
    fig.legend(lines, labels=batch_sizes, title="Batch Size:")
    fig.savefig('../data/graphs/model_comparison.png')
    # fig.savefig('../data/graphs/distribution_rhetorical_roles_balanced.png')

    plt.show()
    # plt.show(block=True)


if __name__ == "__main__":
    # plot_batch_sizes_across_models()
    plot_dataset()
