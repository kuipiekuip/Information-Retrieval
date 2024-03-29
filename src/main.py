from geturl import GetURLs
from preprocessing import download_all_documents, clean_all_documents
from vectorizer import run_query_expansion
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

def expand_query(query_string, isNews : bool):
    processed_query = preprocess_query(query_string)
    # words = set(processed_query.split())
    # query = {processed_query}
    # print(original_query)

    if processed_query not in temp_storage['news' if isNews else 'bing']:
        url_list = GetURLs(processed_query, 20, isNews)
        print(url_list)
        page_list = download_all_documents(url_list=url_list)
        text_list = clean_all_documents(page_list)
        for text in text_list:
            print(len(text))

        output = run_query_expansion(text_list, processed_query)
        write_temporary_result(processed_query, output, isNews)
    else:
        output = temp_storage['news' if isNews else 'bing'][processed_query]

    result = query_string + ' ' + ' '.join(output)
    print(result)
    return result

    # print("==================================================================================")
    # print(output)

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

def create_expanded_queries(original_queries : pd.DataFrame, isNews : bool) -> pd.DataFrame:
    df = original_queries.copy()
    df['query'] = df[['query']].apply(lambda x: expand_query(x.values[0], isNews), axis=1)
    return df

def pre_process_queries(queries_raw: pd.DataFrame):
    queries_raw.rename(columns={'title': 'query'}, inplace=True)
    return queries_raw[['qid', 'query']].copy()

def store_result(df : pd.DataFrame, file_path: str):
    dir = 'topics/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    df.to_csv(dir + file_path, index=False)

def load_result(file_path : str) -> pd.DataFrame:
    dir = 'topics/'
    return pd.read_csv(dir + file_path).astype(object)


if __name__== "__main__" :
    if not pt.started():
        pt.init()
    dataset = pt.datasets.get_dataset('irds:nyt/trec-core-2017')
    original_queries = pre_process_queries(dataset.get_topics())

    temp_storage = load_temporary_result()
    
    bing_queries = create_expanded_queries(original_queries, isNews=False)
    store_result(bing_queries, 'bing.csv')

    bing_news_queries = create_expanded_queries(original_queries, isNews=True)
    store_result(bing_news_queries, 'bing_news.csv')


