import pyterrier as pt
import pandas as pd
from pyterrier.measures import RR, nDCG, MAP

def run_evaluation(dataset):
    # dataset = pt.get_dataset(dataset_name) 

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

# docs = load_dataset('irds/nyt', 'docs')
# for record in docs:
#         print(record)


# dataset = pt.datasets.get_dataset('irds:cord19/fulltext/trec-covid')
# print(dataset.get_topics())
# print(dataset.get_qrels())
# for document in dataset.get_corpus_iter():
    # print(document)


dataset = pt.datasets.get_dataset('irds:nyt/trec-core-2017')
# print(dataset.get_topics())
# print(dataset.get_qrels())
for document in dataset.get_corpus_iter():
    print(document)
    

default, regular, news = run_evaluation(dataset)
print(default)
print(regular)
print(news)

# 3 queries, 1-3 datasets, 1-3 weighting models

# DataFrame['qid','query] -> for each query we find all urls -> for all urls we retrieve the text -> we do the magic -> create expanded -> DataFrrame['qid','query']