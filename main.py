import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import nltk
import string
import re
import gc  # garbage collector to manage RAM usage
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    tweets = pd.read_csv(file_path)
    tweets.rename(columns={'label': 'Label', 'tweet': 'Tweet'}, inplace=True)
    tweets.drop(columns=['id'], inplace=True)
    return tweets

# Function to split the dataset into training and testing sets
def split_dataset(tweets):
    X_train, X_test, y_train, y_test = train_test_split(
        tweets['Tweet'],
        tweets['Label'],
        test_size=0.2,
        stratify=tweets['Label'],
        random_state=1
    )
    return X_train, X_test, y_train, y_test

# Function to clean a tweet
def clean_tweet(tweet):
    twt_tokenizer = TweetTokenizer(strip_handles=True)
    tokens = [token for token in twt_tokenizer.tokenize(tweet)]
    stops = stopwords.words("english") + list(string.punctuation)
    lemmatizer = WordNetLemmatizer()
    tokens_no_hashtag = [re.sub(r'#', '', token) for token in tokens]
    tokens_no_stopwords = [token.lower() for token in tokens_no_hashtag if token.lower() not in stops]
    tokens_no_url = [re.sub(r'http\S+', '', token) for token in tokens_no_stopwords]
    tokens_no_url = [re.sub(r'www\S+', '', token) for token in tokens_no_url]
    tokens_no_extra_space = [re.sub(r'\s\s+', '', token) for token in tokens_no_url]
    tokens_alnum = [token for token in tokens_no_extra_space if token.isalnum()]
    tokens_lemma = [lemmatizer.lemmatize(token) for token in tokens_alnum]
    tokens_final = [token for token in tokens_lemma if len(token) > 1]
    return tokens_final

# Function to preprocess and vectorize the text data
def preprocess_and_vectorize_text(X_train, X_test):
    corpus_train = X_train.apply(lambda x: ' '.join(clean_tweet(x)))

    vectorizer = CountVectorizer()
    X_train_wc = vectorizer.fit_transform(corpus_train)

    corpus_test = X_test.apply(lambda x: ' '.join(clean_tweet(x)))
    X_test_wc = vectorizer.transform(corpus_test)

    return X_train_wc, X_test_wc, vectorizer

# Function to train a Multinomial Naive Bayes model
def train_mnb_model(X_train_wc, y_train):
    clf = MultinomialNB(alpha=1)
    clf.fit(X_train_wc, y_train)
    return clf

# Function to evaluate a classification model
def evaluate_model(clf, X_test_wc, y_test):
    y_pred = clf.predict(X_test_wc)
    accuracy = clf.score(X_test_wc, y_test)
    return y_pred, accuracy

# Function to display the confusion matrix
def display_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Hate", "Hate"])
    disp.plot()
    plt.show()

# Function to calculate classification metrics
def calculate_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, target_names=["Non-Hate", "Hate"])
    return accuracy, classification_rep

# Main function
def main():
    file_path = 'train.csv'
    tweets = load_and_preprocess_data(file_path)
    X_train, X_test, y_train, y_test = split_dataset(tweets)
    X_train_wc, X_test_wc, vectorizer = preprocess_and_vectorize_text(X_train, X_test)
    clf = train_mnb_model(X_train_wc, y_train)
    y_pred, accuracy = evaluate_model(clf, X_test_wc, y_test)
    display_confusion_matrix(y_test, y_pred)
    accuracy, classification_rep = calculate_metrics(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_rep)

if __name__ == "__main__":
    main()
