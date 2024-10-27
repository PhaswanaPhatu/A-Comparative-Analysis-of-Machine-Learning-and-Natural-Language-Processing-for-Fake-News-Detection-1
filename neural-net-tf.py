#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# from getEmbeddings import getEmbeddings
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import scikitplot as skplt
import os



def plot_cmat(y_true, y_pred):
    '''Plotting confusion matrix'''
    from scikitplot.metrics import plot_confusion_matrix
    plot_confusion_matrix(y_true, y_pred, figsize=(8, 6))
    plt.show()


# Read and preprocess the data
if not os.path.isfile('./xtr.npy') or \
        not os.path.isfile('./xte.npy') or \
        not os.path.isfile('./ytr.npy') or \
        not os.path.isfile('./yte.npy'):
    print("No embeddings")
    # xtr, xte, ytr, yte = getEmbeddings("datasets/train.csv")

xtr = np.load('./xtr_shuffled.npy')
xte = np.load('./xte_shuffled.npy')
ytr = np.load('./ytr_shuffled.npy')
yte = np.load('./yte_shuffled.npy')

# Encode labels
ytr_encoded = to_categorical(ytr)
yte_encoded = to_categorical(yte)


def build_model(input_dim, output_dim):
    '''Neural network with regularization and improved optimizer'''
    model = Sequential()
    model.add(Dense(512, input_dim=input_dim, activation='relu', kernel_initializer='he_normal',
                    kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.4))
    model.add(Dense(512, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(output_dim, activation='softmax', kernel_initializer='he_normal'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model


# Split the data
x_train, x_val, y_train, y_val = train_test_split(xtr, ytr_encoded, test_size=0.3, random_state=42)

# Build and train the model
model = build_model(xtr.shape[1], ytr_encoded.shape[1])
model.summary()

# Train the model without early stopping or model checkpoint
history = model.fit(x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=300,
                    batch_size=64,
                    verbose=1)

# Evaluate the model
score = model.evaluate(xte, yte_encoded, verbose=1)
print(f"Accuracy = {score[1] * 100:.2f}%")

# Predict and plot confusion matrix
y_pred_proba = model.predict(xte)
y_pred = np.argmax(y_pred_proba, axis=1)
plot_cmat(yte, y_pred)

# Generate and print classification report
print("\nClassification Report:\n")
print(classification_report(yte, y_pred))