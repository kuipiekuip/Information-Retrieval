from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


# This function also does the tokenization and stop word removal, so no preprocessing is needed
def compute_tfidf_and_select_top_terms(corpus_C, M=100):

    # Initialize the TF-IDF Vectorizer with English stop words
    vectorizer = TfidfVectorizer(stop_words="english")

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


def calculate_custom_weights(corpus):
    """
    Calculate custom weights for terms in each document in the corpus.

    :param corpus: List of documents, where each document is a string.
    :return: A list of dictionaries, each representing the term weights for a document.
    """
    # Initialize CountVectorizer to count term frequencies
    vectorizer = CountVectorizer(stop_words="english")
    tf_matrix = vectorizer.fit_transform(corpus)

    # Total number of terms in the collection
    T = tf_matrix.sum()

    # List to store term weights for each document
    document_term_weights = []

    # Calculate custom weights for each document
    for j in range(tf_matrix.shape[0]):
        # Number of distinct terms in document dj
        DTj = tf_matrix[j].getnnz()

        # Calculate ITF
        itf = np.log(T / DTj)

        # Extract term frequencies for document dj
        tf = tf_matrix[j].toarray().flatten()

        # Calculate custom weights for terms in document dj
        wti_j = tf * itf

        # Map terms to their weights
        terms = vectorizer.get_feature_names_out()
        term_weights = {
            term: wti_j[idx] for idx, term in enumerate(terms) if wti_j[idx] > 0
        }

        document_term_weights.append(term_weights)

    return document_term_weights


def create_term_vectors(document_term_weights):
    """
    Create a unified term vector for each term across all documents.

    :param document_term_weights: List of dictionaries with term weights for each document.
    :return: Dict of numpy arrays representing term vectors.
    """
    # Combine all terms from all documents
    all_terms = set(
        term for doc_weights in document_term_weights for term in doc_weights
    )

    # Initialize vectors
    term_vectors = {term: [] for term in all_terms}

    # Build vectors
    for term in all_terms:
        for doc_weights in document_term_weights:
            weight = doc_weights.get(term, 0)  # Use 0 if term not in document
            term_vectors[term].append(weight)

    # Convert lists to numpy arrays
    for term in term_vectors:
        term_vectors[term] = np.array(term_vectors[term])

    return term_vectors


def compute_cosine_similarities(term_vectors):
    """
    Compute cosine similarities between all pairs of term vectors.

    :param term_vectors: Dict of numpy arrays for each term.
    :return: A matrix of cosine similarity values.
    """
    terms = list(term_vectors.keys())
    vectors = np.array([term_vectors[term] for term in terms])

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(vectors)

    return terms, similarity_matrix


def k_nearest_neighbors(Cexp, terms, similarity_matrix, k, l, r=5):
    """
    Implement the k-Nearest Neighbors algorithm for selecting expansion terms.

    :param Cexp: List of intermediate candidate expansion terms sorted by initial score.
    :param terms: List of all terms in the similarity matrix.
    :param similarity_matrix: Matrix of cosine similarity scores between terms.
    :param k: Number of nearest neighbor candidate expansion terms to return.
    :param l: Number of terms to be dropped in each iteration.
    :param r: Number of iterations.
    :return: Set of iteratively added nearest neighbor candidate expansion terms.
    """
    NN = set()  # Initialize the set of nearest neighbors
    term_index_map = {term: index for index, term in enumerate(terms)}
    # print(term_index_map)
    while r > 0 and Cexp:
        # Select the term with the maximum initial score
        t = Cexp[0][0]
        NN.add(Cexp[0])  # Add t to NN
        Cexp.remove(Cexp[0])  # Remove t from Cexp

        if not Cexp:  # If Cexp is empty, break the loop
            break

        # Sort Cexp w.r.t t using cosine similarity score
        t_index = term_index_map[t]
        Cexp.sort(
            key=lambda term: similarity_matrix[term_index_map[term[0]], t_index],
            reverse=True,
        )

        # Select t again based on updated Cexp
        t = Cexp[0]

        # Drop the l least scoring terms from Cexp
        if l < len(Cexp):
            Cexp = Cexp[:-l]

        r -= 1  # Decrement the iteration counter

    # After iterations, select the top k terms from the remaining Cexp to add to NN
    for term in Cexp[: k - len(NN)]:
        NN.add(term)

    return NN


def sum_product_weights(term_t, term_q, document_term_weights):
    """
    Calculate the sum of the products of weights for two terms across all documents.
    """
    sum_product = 0
    for doc_weights in document_term_weights:
        weight_t = doc_weights.get(term_t, 0)
        weight_q = doc_weights.get(term_q, 0)
        sum_product += weight_t * weight_q
    return sum_product


def correlation_score_single(
    candidate_query_term, original_query, document_term_weights
):
    """
    Calculate the correlation score of a single candidate query term with respect to the original query.
    """
    words = original_query.split()  # Splitting the original query into individual terms
    sum_correlation = sum(
        sum_product_weights(candidate_query_term, reference_term, document_term_weights)
        for reference_term in words
    )
    return sum_correlation / len(words) if words else 0


def sort_by_correlation(
    candidate_query_terms_set, original_query, document_term_weights
):
    """
    Sort the candidate query terms by their correlation scores in descending order.
    """
    # Calculating correlation scores for each candidate term
    correlation_scores = [
        (term, correlation_score_single(term, original_query, document_term_weights))
        for term in candidate_query_terms_set
    ]

    # Sorting terms by their scores
    sorted_terms = sorted(correlation_scores, key=lambda x: x[1], reverse=True)
    sorted_terms_dict = {term: score for term, score in sorted_terms}

    return sorted_terms_dict


def run_query_expansion(corpus, original_query, M=100, k=20, l=100, r=5):
    # Compute the top M terms using TF-IDF
    top_m_terms = compute_tfidf_and_select_top_terms(corpus, M)
    # Calculate custom weights for terms in each document
    document_term_weights = calculate_custom_weights(corpus)
    # Create term vectors for all terms in the document collection
    term_vectors = create_term_vectors(document_term_weights)
    # Compute cosine similarities between all pairs of term vectors
    terms, similarity_matrix = compute_cosine_similarities(term_vectors)
    # Run the k-Nearest Neighbors algorithm to select expansion terms
    knn_scores = k_nearest_neighbors(top_m_terms, terms, similarity_matrix, k, l, r)
    # Extract the terms from the k-NN scores
    knn_terms = {term for term, score in knn_scores}

    # Perform query expansion using the k-NN terms and the original query terms
    sorted_candidate_terms = sort_by_correlation(
        knn_terms, original_query, document_term_weights
    )
    return sorted_candidate_terms


if __name__ == "__main__":
    # Define the corpus
    documents = [
        "Alice's Adventures in Wonderland is an 1865 novel written by English author Lewis Carroll. It tells of a young girl named Alice, who falls through a rabbit hole into a subterranean fantasy world populated by peculiar, anthropomorphic creatures. It is considered to be one of the best examples of the literary nonsense genre. The tale plays with logic, giving the story lasting popularity with adults as well as with children.",
        "Pride and Prejudice is a romantic novel of manners written by Jane Austen in 1813. The novel follows the character development of Elizabeth Bennet, the dynamic protagonist of the book who learns about the repercussions of hasty judgments and comes to appreciate the difference between superficial goodness and actual goodness. The novel is set in rural England in the early 19th century and it charts the emotional development of the protagonist who learns the error of making hasty judgments and comes to appreciate the difference between the superficial and the essential.",
        "Moby-Dick; or, The Whale is an 1851 novel by American writer Herman Melville. The book is the sailor Ishmael's narrative of the obsessive quest of Ahab, captain of the whaling ship Pequod, for revenge on Moby Dick, the giant white sperm whale that on the ship's previous voyage bit off Ahab's leg at the knee.",
    ]

    print(run_query_expansion(corpus=documents))