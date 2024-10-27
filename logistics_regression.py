import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


xtr = np.load('./xtr_shuffled.npy')
xte = np.load('./xte_shuffled.npy')
ytr = np.load('./ytr_shuffled.npy')
yte = np.load('./yte_shuffled.npy')

def preprocess_data(xtr, xte):
    scaler = MinMaxScaler()
    xtr_scaled = scaler.fit_transform(xtr)
    xte_scaled = scaler.transform(xte)
    return xtr_scaled, xte_scaled

def preprocess_data_with_smote(xtr, ytr):
    smote = SMOTE(random_state=42)
    xtr_resampled, ytr_resampled = smote.fit_resample(xtr, ytr)
    return xtr_resampled, ytr_resampled

xtr, xte = preprocess_data(xtr, xte)
xtr, ytr = preprocess_data_with_smote(xtr, ytr)

def train_and_evaluate_logistic_regression(xtr, xte, ytr, yte):
    clf = LogisticRegression()
    clf.fit(xtr, ytr)
    y_pred = clf.predict(xte)
    accuracy = accuracy_score(yte, y_pred) * 100
    print(f"Accuracy = {accuracy:.2f}%")
    print("Classification Report:")
    print(classification_report(yte, y_pred))
    plot_confusion_matrix(yte, y_pred)

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

def main():
    train_and_evaluate_logistic_regression(xtr, xte, ytr, yte)

if __name__ == '__main__':
    main()


