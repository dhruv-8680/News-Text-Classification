import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import re

from nltk.corpus import stopwords    
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, precision_score, recall_score
from sklearn.metrics import roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\.{2,}', ' ', text).strip()
    
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    cleaned_text = ' '.join(tokens)
    cleaned_text = re.sub(r'\. ', ' ', cleaned_text).strip()
    return cleaned_text





def frequencyPlot(df, column_name, percent = 95, total_top_class = 20):
    df = df.astype(str).copy()
    percent_str = str(percent)
    percent /= 100
    _cumsum = df.value_counts(normalize=True).cumsum()
    
    _freq = df[df.isin(_cumsum[_cumsum<percent].index)].value_counts(normalize=True)\
                        .reset_index().rename(columns={'index':column_name, 
                                                       "proportion":"count"})[:total_top_class]
    _freq_2 = df.value_counts(normalize=True)\
                        .reset_index().rename(columns={'index':column_name, 
                                                       "proportion":"count"})[:total_top_class]
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(list(range(0, df.unique().shape[0])), _cumsum)
    plt.title("Cumulative Frequency "+column_name.title()+" | "+\
              percent_str+"% of data explained by " + str(sum(_cumsum<percent))+" / "+str(df.unique().shape[0]))
    plt.hlines(percent,0, _cumsum.shape[0],'k', '--', label=percent_str+"% threshold")
    plt.vlines(sum(_cumsum<percent),0, 1,'r', '--', label="Total Account "+str(sum(_cumsum<percent)))
    plt.xlabel("Total "+column_name.title())
    plt.legend()

    plt.subplot(1,2,2)
    sns.barplot(x=_freq_2[column_name], y=_freq_2["count"])
    plt.title("Frequency "+column_name.title()+" | Top "+str(total_top_class)+" covers " +\
              str((round(_freq_2["count"].sum(),5)*100))+"%")
    plt.xlabel(column_name)
    plt.xticks(rotation=90)
    plt.show()
    
    boolean_index = df.isin(_cumsum[:int(sum(_cumsum<percent))].reset_index()[column_name])
    column_values = df[boolean_index]
    total_data_size = sum(boolean_index)
    
    return (boolean_index, column_values, total_data_size)

def check_metric(y_true, y_pred, pred_prob, y_test_true=None, y_test_pred=None, test_pred_prob=None, title=None):
    if y_test_pred is None:
        f1_ = f1_score(y_true, y_pred)
        roc_auc_ = roc_auc_score(y_true, y_pred)
        precision_ = precision_score(y_true, y_pred)
        recall_ = recall_score(y_true, y_pred)
        print("F1 Score: ",f1_)
        print("ROC AUC Score: ",roc_auc_)
        print("Precision Score: ",precision_)
        print("Recall Score: ",recall_)
    else:
        f1_ = f1_score(y_true, y_pred) 
        roc_auc_ = roc_auc_score(y_true, y_pred)
        precision_ = precision_score(y_true, y_pred)
        recall_ = recall_score(y_true, y_pred)

        test_f1_ = f1_score(y_test_true, y_test_pred) 
        test_roc_auc_ = roc_auc_score(y_test_true, y_test_pred)
        test_precision_ = precision_score(y_test_true, y_test_pred)
        test_recall_ = recall_score(y_test_true, y_test_pred)

        print("Test F1 Score: ",test_f1_, " | Train F1 Score: ",f1_)
        print("Test ROC AUC Score: ",test_roc_auc_, " | Train ROC AUC Score: ",roc_auc_)
        print("Test Precision Score: ",test_precision_, " | Train Precision Score: ",precision_)
        print("Test Recall Score: ",test_recall_, " | Train Recall Score: ",recall_)
    
    plt.figure(figsize=(7, 7))
    
    if y_test_true is not None:
        test_fpr, test_tpr, test_thresh = roc_curve(y_test_true, test_pred_prob[:,1])
        test_roc_auc_score = auc(test_fpr, test_tpr)
        plt.plot(test_fpr, test_tpr, label="Test auc_score: "+str(round(test_roc_auc_score, 4)))
        print("Test Threshold: ",test_thresh[np.argmax(test_tpr-test_fpr)])

    train_fpr, train_tpr, train_thresh = roc_curve(y_true, pred_prob[:,1])
    train_roc_auc_score = auc(train_fpr, train_tpr)
    plt.plot(train_fpr, train_tpr, label="Train auc_score: "+str(round(train_roc_auc_score, 4)))
    print("Train Threshold: ",train_thresh[np.argmax(train_tpr-train_fpr)])

    plt.plot([0,1], [0,1], "--k")
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.tight_layout()
    plt.show()