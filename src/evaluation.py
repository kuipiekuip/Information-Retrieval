import pyterrier as pt
import pandas as pd
from pyterrier.measures import RR, nDCG, MAP

def run_evaluation(topics : pd.DataFrame):
    dataset = pt.datasets.get_dataset('irds:beir/trec-covid')
    index_ref = pt.IndexRef.of('./indices/cord19_fulltext')
    return evaluation(dataset, index_ref, topics)

def evaluation(dataset, index_ref, topics : pd.DataFrame) -> pd.DataFrame:
    tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")

    return pt.Experiment(
        [tfidf, bm25],
        topics,
        dataset.get_qrels(),
        eval_metrics=[nDCG @ 5, nDCG @ 10, nDCG @ 100],
    )