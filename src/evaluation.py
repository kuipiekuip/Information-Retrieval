import pyterrier as pt
import pandas as pd
from pyterrier.measures import RR, nDCG, MAP

def run_evaluation():
    dataset = pt.datasets.get_dataset('irds:nyt/trec-core-2017')
    queries = dataset.get_topics()
    queries.rename(columns={'title': 'query'}, inplace=True)
    print(queries.dtypes)
    index_ref = pt.IndexRef.of('./indices/nyt')

    # dummy_df = default_queries.copy() # This is where we read in our own query expansions
    # dummy_df['query'] = 'some query'

    # expanded_regular_queries = dummy_df
    # expanded_news_queries = dummy_df

    result_default = evaluation(dataset, index_ref, queries)
    # result_expanded_regular = evaluation(dataset, expanded_regular_queries)
    # result_expanded_news = evaluation(dataset, expanded_news_queries)

    return result_default#, result_expanded_regular, result_expanded_news

def evaluation(dataset, index_ref, topics : pd.DataFrame) -> pd.DataFrame:
    tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

    return pt.Experiment(
        [tfidf, bm25],
        topics,
        dataset.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 20, MAP],
    )

if not pt.started():
    pt.init()


result = run_evaluation()
print(result)