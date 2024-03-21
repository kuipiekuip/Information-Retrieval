from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer



def tokenize_document(document):
    vectorizer = TfidfVectorizer(lowercase=True)
    tokenized_array = vectorizer.fit_transform([document]).toarray()
    tokens = vectorizer.get_feature_names_out()

    return tokens

def tokenize_document_freq(document):
    vectorizer = CountVectorizer(lowercase=True)
    tokenized_array = vectorizer.fit_transform([document]).toarray()
    tokens = vectorizer.get_feature_names_out()
    frequency = tokenized_array[0]
    token_frequency = dict(zip(tokens, frequency))

    return token_frequency

def get_corpus(documents):
    # Join all documents into a single string
    corpus = ' '.join(documents)
    return corpus

