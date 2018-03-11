#Import package
import nltk
from collections import Counter
import pandas as pd
import string
import numpy as np
import sklearn

#Download NLTK's stopwords list and WordNetLemmatizer.
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stopwords = nltk.corpus.stopwords.words('english')


def learn():

    def process(text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
        """ Normalizes case and handles punctuation
        Inputs:
            text: str: raw text
            lemmatizer: an instance of a class implementing the lemmatize() method
                        (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
        Outputs:
            list(str): tokenized text
        """
        text = text.lower()
        text = " " + text + " "

        text = text.replace("'s", " ")

        text = text.replace("'", "")

        t = string.punctuation.replace("'", "")
        trans = str.maketrans(t, " " * len(t))
        text = text.translate(trans)

        # tokenize
        tokens = nltk.word_tokenize(text)

        ltokens = []

        for token in tokens:
            try:
                ltokens.append(lemmatizer.lemmatize(token))
            except:
                ltokens = ltokens

        return ltokens

    def process_all(df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
        """ process all text in the dataframe using process_text() function.
        Inputs
            df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
            lemmatizer: an instance of a class implementing the lemmatize() method
                        (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
        Outputs
            pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                            the output from process_text() function. Other columns are unaffected.
        """
        df1 = df.copy()
        df1['text'] = df1['text'].apply(process, lemmatizer)
        return df1

    def get_rare_words(processed_tweets):
        """ use the word count information across all tweets in training data to come up with a feature list
        Inputs:
            processed_tweets: pd.DataFrame: the output of process_all() function
        Outputs:
            list(str): list of rare words, sorted alphabetically.
        """
        df_list = processed_tweets['text'].tolist()
        c = Counter([x for word in df_list for x in word])
        list_all = c.items()
        one_element = []
        for i in list_all:
            if i[1] == 1:
                one_element.append(i[0])
        return sorted(one_element)

    def create_features(processed_tweets, rare_words):
        """ creates the feature matrix using the processed tweet text
        Inputs:
            tweets: pd.DataFrame: tweets read from train/test csv file, containing the column 'text'
            rare_words: list(str): one of the outputs of get_feature_and_rare_words() function
        Outputs:
            sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
                                                    we need this to tranform test tweets in the same way as train tweets
            scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
        """
        df = processed_tweets.copy()
        df['text'] = df['text'].apply(lambda x: str(' '.join(x)))
        stop = set(rare_words).union(stopwords)
        tfidf = sklearn.feature_extraction.text.TfidfVectorizer(stop_words=stop)
        X = tfidf.fit_transform(df['text'])

        return (tfidf, X)

    def create_labels(processed_tweets):
        """ creates the class labels from screen_name
        Inputs:
            tweets: pd.DataFrame: tweets read from train file, containing the column 'screen_name'
        Outputs:
            numpy.ndarray(int): dense binary numpy array of class labels
        """
        df = processed_tweets.copy()
        df['label'] = 1
        #The array elements can be modified according the user tweets need.
        #Here is a list of republican figures we thing they use words that tend to classify them as republican.
        df.loc[df['screen_name'].isin(['realDonaldTrump', 'mike_pence', 'GOP']), 'label'] = 0

        return np.array(df['label'])

    def learn_classifier(X_train, y_train, kernel='best'):
        """ learns a classifier from the input features and labels using the kernel function supplied
        Inputs:
            X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features, output of create_features_and_labels()
            y_train: numpy.ndarray(int): dense binary vector of class labels, output of create_features_and_labels()
            kernel: str: kernel function to be used with classifier. [best|linear|poly|rbf|sigmoid]
                        if 'best' is supplied, reset the kernel parameter to the value you have determined to be the best
        Outputs:
            sklearn.svm.classes.SVC: classifier learnt from data
        """
        if kernel == 'best':
            #After evaluating the different classifier (linear, poly, rbf, sigmoid),
            # the linear classifier has the best mode.
            kernel = 'linear'
        s = sklearn.svm.SVC(kernel=kernel)
        return s.fit(X_train, y_train)

    def evaluate_classifier(classifier, X_validation, y_validation):
        """ evaluates a classifier based on a supplied validation data
        Inputs:
            classifier: sklearn.svm.classes.SVC: classifer to evaluate
            X_train: scipy.sparse.csr.csr_matrix: sparse matrix of features
            y_train: numpy.ndarray(int): dense binary vector of class labels
        Outputs:
            double: accuracy of classifier on the validation data
        """
        clf = classifier.predict(X_validation)
        return sklearn.metrics.accuracy_score(clf,  y_validation)

    def classify_tweets(tfidf, classifier, unlabeled_tweets):
        """ predicts class labels for raw tweet text
        Inputs:
            tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
            classifier: sklearn.svm.classes.SVC: classifier learnt
            unlabeled_tweets: pd.DataFrame: tweets read from tweets_test.csv
        Outputs:
            numpy.ndarray(int): dense binary vector of class labels for unlabeled tweets
        """
        df = pd.DataFrame(unlabeled_tweets.copy())
        df = process_all(df)
        df['text'] = df['text'].apply(lambda x: str(" ".join(x)))
        X = tfidf.transform(df['text'])
        y_pred = classifier.predict(X)
        return y_pred