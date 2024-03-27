import pyterrier as pt
import pandas as pd
from pyterrier.measures import RR, nDCG, MAP


def run_evaluation(dataset_name : str):
    dataset = pt.get_dataset(dataset_name) 
    default_queries = dataset.get_topics()
    dummy_df = default_queries.copy() # This is where we read in our own query expansions
    dummy_df['query'] = 'some query'
    expanded_regular_queries = dummy_df
    expanded_news_queries = dummy_df

    result_default = evaluation(dataset, default_queries)
    result_expanded_regular = evaluation(dataset, expanded_regular_queries)
    result_expanded_news = evaluation(dataset, expanded_news_queries)

    return result_default, result_expanded_regular, result_expanded_news

def evaluation(dataset, topics : pd.DataFrame) -> pd.DataFrame:
    index = dataset.get_index(variant="terrier_stemmed")

    tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    return pt.Experiment(
        [tfidf, bm25],
        topics,
        dataset.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 20, MAP],
    )

if not pt.started():
    pt.init()


default, regular, news = run_evaluation("vaswani")
print(default)
print(regular)
print(news)