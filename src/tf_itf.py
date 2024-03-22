import math
import nltk
from helper import tokenize_document, get_corpus, tokenize_document_freq
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import time



def get_tf_id(corpus_C, M= 50):
    vectorizer = TfidfVectorizer(stop_words='english')

    # Fit the model and transform the documents
    X = vectorizer.fit_transform(corpus_C)
    # Get feature names to use as terms
    terms = vectorizer.get_feature_names_out()
    # Sum TF-IDF scores for each term across all documents
    summed_tfidf = np.sum(X, axis=0)
    # Get the scores into a readable format
    scores = np.ravel(summed_tfidf)
    # Pair terms with their scores
    term_scores = list(zip(terms, scores))
    # Sort terms by their scores descending
    sorted_term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)
    # Select top M terms
    top_m_terms = sorted_term_scores[:M]

    return top_m_terms
    # # Initialize the TF-IDF Vectorizer with English stop words
    # vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    # tfidf_matrix = vectorizer.fit_transform(documents)
    # feature_names = vectorizer.get_feature_names_out()
    # term_scores = {}

    # for col in range(tfidf_matrix.shape[1]):
    #     term = feature_names[col] 
    #     score = tfidf_matrix[:, col].sum()  
    #     term_scores[term] = score

    # sorted_term_scores = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)

    # return sorted_term_scores

def get_tf2(term, document):
    return document.count(term)

def get_tf(term, document):
    return tokenize_document_freq(document).get(term, 0)

def get_itf(document, corpus):
    return math.log(len(tokenize_document(corpus)) / len(tokenize_document(document)))

def get_tf_itf(term, document, corpus):
    return get_tf(term, document)*get_itf(document, corpus)

def get_similarity(term1, term2, documents):
    corpus = get_corpus(documents)

    numerator = 0.0
    denumerator1 = 0.0
    denumerator2 = 0.0

    for i in range(len(documents)):
        weight1 = get_tf_itf(term1, documents[i], corpus)
        weight2 = get_tf_itf(term2, documents[i], corpus)

        numerator += weight1 * weight2
        denumerator1 += weight1 ** 2
        denumerator2 += weight2 ** 2


    denumerator = math.sqrt(denumerator1 * denumerator2)


    return numerator, denumerator

def cosine_sort(term, candidates, documents):
    term_scores = []
    for candidate_term in candidates:

        numerator, denumarator = get_similarity(term, candidate_term[0], documents)
        term_scores.append((candidate_term[0], numerator/denumarator))

    sorted_term_scores = sorted(term_scores, key=lambda x: x[1], reverse=True)
    
    return sorted_term_scores


def get_kNN(candidates, k, l, documents, r = 5):
    NN = []
    t = candidates[0][0]
    r2 = k-r

    while r > 0:
        NN.append(t)
        candidates.pop(0)
        candidates = cosine_sort(t, candidates, documents)

        for _ in range(l):
            if not candidates:  #check if list is not empty
                candidates.pop()

        t = candidates[0][0]
        r -= 1

    for i in range(r2):
        NN.append(candidates[i])

    return NN



def main():
    documents = [
        "Alice's Adventures in Wonderland is an 1865 novel written by English author Lewis Carroll. It tells of a young girl named Alice, who falls through a rabbit hole into a subterranean fantasy world populated by peculiar, anthropomorphic creatures. It is considered to be one of the best examples of the literary nonsense genre. The tale plays with logic, giving the story lasting popularity with adults as well as with children.",
        "Pride and Prejudice is a romantic novel of manners written by Jane Austen in 1813. The novel follows the character development of Elizabeth Bennet, the dynamic protagonist of the book who learns about the repercussions of hasty judgments and comes to appreciate the difference between superficial goodness and actual goodness. The novel is set in rural England in the early 19th century and it charts the emotional development of the protagonist who learns the error of making hasty judgments and comes to appreciate the difference between the superficial and the essential.",
        "Moby-Dick; or, The Whale is an 1851 novel by American writer Herman Melville. The book is the sailor Ishmael's narrative of the obsessive quest of Ahab, captain of the whaling ship Pequod, for revenge on Moby Dick, the giant white sperm whale that on the ship's previous voyage bit off Ahab's leg at the knee.",
    ]

    corpus = get_corpus(documents)

    print(get_kNN(get_tf_id(documents), 40, 20, documents))


if __name__ == '__main__':
    
    documents = [
        "Alice's Adventures in Wonderland is an 1865 novel written by English author Lewis Carroll. It tells of a young girl named Alice, who falls through a rabbit hole into a subterranean fantasy world populated by peculiar, anthropomorphic creatures. It is considered to be one of the best examples of the literary nonsense genre. The tale plays with logic, giving the story lasting popularity with adults as well as with children.",
        "Pride and Prejudice is a romantic novel of manners written by Jane Austen in 1813. The novel follows the character development of Elizabeth Bennet, the dynamic protagonist of the book who learns about the repercussions of hasty judgments and comes to appreciate the difference between superficial goodness and actual goodness. The novel is set in rural England in the early 19th century and it charts the emotional development of the protagonist who learns the error of making hasty judgments and comes to appreciate the difference between the superficial and the essential.",
        "Moby-Dick; or, The Whale is an 1851 novel by American writer Herman Melville. The book is the sailor Ishmael's narrative of the obsessive quest of Ahab, captain of the whaling ship Pequod, for revenge on Moby Dick, the giant white sperm whale that on the ship's previous voyage bit off Ahab's leg at the knee.",
    ]
    alice_text = "Your Alice novel goes here."
    novel_text = "Your Novel text goes here."
    documents = [alice_text, novel_text]
    # t0 = time.time()
    # main()
    # t1 = time.time()

    # total = t1-t0
    # print(total)
    numerator, denumerator = get_similarity("alice", "novel", documents)
    print(numerator/denumerator)