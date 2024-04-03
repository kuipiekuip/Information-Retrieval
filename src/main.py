from geturl import GetURLs
from preprocessing import download_all_documents, clean_all_documents
from vectorizer import run_query_expansion
from evaluation import run_evaluation
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pyterrier as pt
import pandas as pd
import os
import json

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

temp_storage = {"bing" : {}, "news" : {}}



def preprocess_query(query):
    words = word_tokenize(query)
    # Remove stopwords
    filtered_query = [word for word in words if not word.lower() in stop_words]
    # Rejoin the filtered words into a string
    filtered_text = ' '.join(filtered_query)
    return filtered_text

def expand_query(query_string, isNews : bool, numTerms : int):
    processed_query = preprocess_query(query_string)

    if processed_query not in temp_storage['news' if isNews else 'bing']:
        url_list = GetURLs(processed_query, 20, isNews)
        print(url_list)
        page_list = download_all_documents(url_list=url_list)
        text_list = clean_all_documents(page_list)

        output = run_query_expansion(text_list, processed_query, k=numTerms)
        write_temporary_result(processed_query, output, isNews)
    else:
        output = temp_storage['news' if isNews else 'bing'][processed_query]

    result = query_string + ' ' + ' '.join([*output])

    return result

def write_temporary_result(query : str, output, isNews : bool):
    temp_storage['news' if isNews else 'bing'][query] = output
    with open('temp_results.json', 'w') as file:
        json.dump(temp_storage, file)

def load_temporary_result():
    if not os.path.exists('temp_results.json'):
        return {"bing" : {}, "news" : {}}
    with open('temp_results.json', 'r') as file:
        temp_storage = json.load(file)
    return temp_storage

def create_expanded_queries(original_queries : pd.DataFrame, isNews : bool, numTerms : int) -> pd.DataFrame:
    df = original_queries.copy()
    df['query'] = df[['query']].apply(lambda x: expand_query(x.values[0], isNews, numTerms), axis=1)
    return df

def pre_process_queries(queries_raw: pd.DataFrame):
    return queries_raw[['qid', 'query']].copy()

if __name__== "__main__" :
    if not pt.started():
        pt.init()

    # Retrieve the original queries
    dataset = pt.datasets.get_dataset('irds:beir/trec-covid')
    original_queries = pre_process_queries(dataset.get_topics())

    # Load the previous results
    temp_storage = load_temporary_result()
    
    # Create the expanded queries for Bing Search
    bing_queries = create_expanded_queries(original_queries, isNews=False, numTerms = 5)

    # Create the expanded queries for News Search
    bing_news_queries = create_expanded_queries(original_queries, isNews=True, numTerms = 5)

    # Run the evaluation
    original_result = run_evaluation(original_queries)
    bing_result = run_evaluation(bing_queries)
    bing_news_result = run_evaluation(bing_news_queries)

    print(original_result)
    print(bing_result)
    print(bing_news_result)

