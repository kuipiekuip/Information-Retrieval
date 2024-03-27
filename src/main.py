from geturl import GetURLs
from preprocessing import download_all_documents, clean_all_documents
from vectorizer import run_query_expansion
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_query(query):
    words = word_tokenize(query)
    # Remove stopwords
    filtered_query = [word for word in words if not word.lower() in stop_words]
    # Rejoin the filtered words into a string
    filtered_text = ' '.join(filtered_query)
    return filtered_text

def main(query_string):
    original_query = {query_string}
    url_list = GetURLs(query_string, 5, False)
    print(url_list)
    page_list = download_all_documents(url_list=url_list)
    text_list = clean_all_documents(page_list)
    for text in text_list:
        print(len(text))
    output = run_query_expansion(text_list, original_query)

    print("==================================================================================")
    print(output)


if __name__== "__main__" :
    # is it correct that the query_string is just a string? Or should it be multiple strings?
    query_string = ' a racket sport'
    query_string = preprocess_query(query_string)
    print(query_string)
    words = set(query_string.split())
    main(query_string=query_string)
    
