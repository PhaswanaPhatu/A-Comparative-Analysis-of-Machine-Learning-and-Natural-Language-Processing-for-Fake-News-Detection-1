import numpy as np
import re
import string
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def text_clean(text):
    # Remove non-letter and non-number characters
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()

    # Remove stopwords and apply lemmatization
    stops = set(stopwords.words("english"))
    text = [lemmatizer.lemmatize(word) for word in text if word not in stops]

    return " ".join(text)


def cleanup(text):
    text = text_clean(text)
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text


def construct_tagged_documents(data):
    """
    Construct TaggedDocument for Doc2Vec model.
    """
    return [TaggedDocument(utils.to_unicode(row).split(), [f'Text_{index}']) for index, row in data.items()]


def add_custom_features(data):
    """
    Add custom features like text length, punctuation count, and digit count.
    """
    data['text_len'] = data['text'].apply(lambda x: len(x.split()))
    data['punctuation_count'] = data['text'].apply(lambda x: sum([1 for char in x if char in string.punctuation]))
    data['digit_count'] = data['text'].apply(lambda x: sum([1 for char in x if char.isdigit()]))
    return data


def get_tfidf_features(text_data, max_features=5000):

    tfidf_vectorizer = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(text_data)
    return tfidf_matrix


def handle_class_imbalance(data):

    # Separate the majority and minority classes
    majority_class = data[data['label'] == 0]
    minority_class = data[data['label'] == 1]

    # Upsample the minority class
    minority_upsampled = resample(minority_class,
                                  replace=True,  # Sample with replacement
                                  n_samples=len(majority_class),  # Match the majority class size
                                  random_state=42)

    # Combine majority and upsampled minority classes
    balanced_data = pd.concat([majority_class, minority_upsampled])

    return balanced_data.sample(frac=1).reset_index(drop=True)  # Shuffle the data


def get_embeddings(path, vector_dimension=300):
    """
    Generate Doc2Vec and custom feature-based training and testing data.
    """
    # Load dataset
    data = pd.read_csv(path)

    # Drop rows with missing text values
    data = data.dropna(subset=['text']).reset_index(drop=True)

    # Clean the text data
    data['text'] = data['text'].apply(cleanup)

    # Handle class imbalance if necessary
    data = handle_class_imbalance(data)

    # Add custom features
    data = add_custom_features(data)

    # Construct TaggedDocument for Doc2Vec
    tagged_documents = construct_tagged_documents(data['text'])
    labels = data['label'].values

    # Train Doc2Vec model
    model = Doc2Vec(
        vector_size=vector_dimension,
        window=5,
        min_count=2,
        workers=8,
        epochs=100,
        dm=1,  # Distributed Memory (better for capturing context)
        hs=0,  # Negative sampling
        negative=10,
        seed=42
    )
    model.build_vocab(tagged_documents)
    model.train(tagged_documents, total_examples=model.corpus_count, epochs=model.epochs)

    # TF-IDF Features
    tfidf_matrix = get_tfidf_features(data['text'], max_features=5000)

    # Prepare train and test splits
    train_size = int(0.7 * len(data))
    test_size = len(data) - train_size

    # Initialize arrays for Doc2Vec embeddings and labels
    train_arrays = np.zeros((train_size, vector_dimension))
    test_arrays = np.zeros((test_size, vector_dimension))
    train_labels = np.zeros(train_size)
    test_labels = np.zeros(test_size)

    # Fill arrays with Doc2Vec embeddings
    for i in range(train_size):
        train_arrays[i] = model.dv[f'Text_{i}']
        train_labels[i] = labels[i]

    for i in range(test_size):
        test_arrays[i] = model.dv[f'Text_{train_size + i}']
        test_labels[i] = labels[train_size + i]

    # Convert TF-IDF features to dense arrays for concatenation
    tfidf_train = tfidf_matrix[:train_size].toarray()
    tfidf_test = tfidf_matrix[train_size:].toarray()

    # Concatenate Doc2Vec and TF-IDF features
    X_train = np.hstack((train_arrays, tfidf_train))
    X_test = np.hstack((test_arrays, tfidf_test))

    return X_train, X_test, train_labels, test_labels


# Usage
X_train, X_test, y_train, y_test = get_embeddings('train.csv')

# Save the processed data
np.save('xtr_shuffled.npy', X_train)
np.save('xte_shuffled.npy', X_test)
np.save('ytr_shuffled.npy', y_train)
np.save('yte_shuffled.npy', y_test)