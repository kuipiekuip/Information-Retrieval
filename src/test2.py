from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample texts
alice_text = "Your Alice text goes here."
novel_text = "Your Novel text goes here."

# Creating a TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# To vectorize texts, they need to be in a list where each element is a text.
texts = [alice_text, novel_text]

# Fit and transform the texts
tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

# Calculate the cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])

print(f"Cosine similarity between 'Alice' and 'Novel': {cosine_sim[0][0]}")
