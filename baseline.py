"""Accepts a Scikit-learn (sklearn in code) model object, loads data to train and evaluate this model
by Jason Liu"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Protocol
from numpy.typing import NDArray
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class SklearnModel(Protocol):
    """Simulates the interface of Scikit-learn models"""
    def fit(self, incoming_train, labels_train, sample_weight=None): 
        """update model parameters using incoming training examples and label"""
        ... # the purpose of this class is to 
    def predict(self, incoming_test) -> NDArray: 
        """make prediction from testing data"""
        ...
    def score(self, incoming_test, labels_test, sample_weight=None) -> NDArray: 
        """get average accuracy from testing examples and label"""
        ...
    def predict_proba(self, labels_test : NDArray) -> NDArray:
        """get probability for each example being categorized to each class""" 
        ...
def run_baseline(train_example, train_label,
    test_example, test_label,                                     
    sklearn_classifier : SklearnModel, 
    outcome_names : list[str], all_outcomes_data_location : str, number_of_folds : int=5):
    """Accepts a sklearn model, loading data to train and evaluate it \n
    Inputs:
        0 data used for training\n
        1 ```sklearn_classifier``` (SkleranModel): a Scikit-learn object (initialized with parameters)\n
        2 ```outcome_names``` (list[str]): list of readable names of data ('outcomes' in this project)\t
        \tit's this function's job to use these readable names to find the folder address of said data\n
        3 ```all_outcomes_data_location``` (str): the 'main' data folder containing all the data\n
        \tsince this won't change, it's reasonable to give the argument a default value\n
        4 ```number_of_folds``` (int): number of sets the cross-validation process splits the data into"""
    sklearn_classifier = make_pipeline(StandardScaler(), sklearn_classifier)
    mean_accuracy_of_test_set_list = [] # mean from ONE cross validation fold!
    area_under_receiver_operating_characteristic_list = []
    average_precision_list = []
    for fold in range(number_of_folds):
        sklearn_classifier.fit(train_example, train_label)
        # preformance metrics below! Some metrics use scores, not predictions!
        mean_accuracy_of_test_set = sklearn_classifier.score(test_example, test_label)
        mean_accuracy_of_test_set_list.append(mean_accuracy_of_test_set)
        scores_test = sklearn_classifier.predict_proba(test_example)[:, 1]
        # print(f"scores_test shape: {scores_test.shape}")
        ROC_AUC = roc_auc_score(y_true=test_label, y_score=scores_test)
        area_under_receiver_operating_characteristic_list.append(ROC_AUC)
        average_precision = average_precision_score(y_true=test_label, y_score=scores_test)
        average_precision_list.append(average_precision)
        print(mean_accuracy_of_test_set)
    mean_accuracy = np.mean(mean_accuracy_of_test_set_list)
    std_accuracy = np.std(mean_accuracy_of_test_set_list)
    mean_roc_auc = np.mean(area_under_receiver_operating_characteristic_list)
    std_roc_auc = np.std(area_under_receiver_operating_characteristic_list)
    mean_average_precision = np.mean(average_precision_list)
    std_average_precision = np.std(average_precision_list)
    print(f"\taccuracy:   mean {mean_accuracy:.4f}",
            f"\n\t          std  {std_accuracy:.4f}",
            f"\n\tROC AUC:  mean {mean_roc_auc:.4f}",
            f"\n\t          std  {std_roc_auc:.4f}",
            f"\n\taverage precision: mean {mean_average_precision:.4f}",
            f"\n\t                   std  {std_average_precision:.4f}")
        
