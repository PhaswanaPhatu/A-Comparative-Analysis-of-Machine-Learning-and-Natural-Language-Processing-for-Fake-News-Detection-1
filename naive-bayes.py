import numpy as np
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data
if not os.path.isfile('./xtr_shuffled.npy') or \
        not os.path.isfile('./xte_shuffled.npy') or \
        not os.path.isfile('./ytr_shuffled.npy') or \
        not os.path.isfile('./yte_shuffled.npy'):
    print("No embeddings found. Please generate embeddings first.")

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


def train_and_evaluate_naive_bayes(xtr, xte, ytr, yte):
    # Create and train the Naive Bayes classifier
    clf = GaussianNB()
    clf.fit(xtr, ytr)

    # Make predictions
    y_pred = clf.predict(xte)

    # Calculate accuracy
    accuracy = accuracy_score(yte, y_pred) * 100
    print(f"Accuracy = {accuracy:.2f}%")

    # Print classification report
    print("Classification Report:")
    print(classification_report(yte, y_pred))

    # Plot confusion matrix
    plot_confusion_matrix(yte, y_pred)


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Generate and print classification report
    print("\nClassification Report:\n")
    print(classification_report(yte, y_pred))


def main():
    train_and_evaluate_naive_bayes(xtr, xte, ytr, yte)


if __name__ == '__main__':
    main()