import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

corpus_C = [
    "the quick brown fox jumps over the lazy dog",
    "never jump over the lazy dog quickly",
    "bright brown fox jumps over the quick dog"
]

def preprocess_corpus(corpus):
    tokenized_corpus = []
    for document in corpus:
        tokens = word_tokenize(document)
        filtered_tokens = [word for word in tokens if word not in stop_words]
        tokenized_corpus.extend(filtered_tokens)
    return tokenized_corpus



corpus_Cf = preprocess_corpus(corpus_C)
print(corpus_Cf)
print(set(corpus_Cf))
