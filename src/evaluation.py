import pyterrier as pt
from pyterrier.measures import RR, nDCG, MAP


def evaluation(dataset_name : str):
    dataset = pt.get_dataset(dataset_name)
    index = dataset.get_index(variant="terrier_stemmed")

    # tfidf = pt.BatchRetrieve(index, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(index, wmodel="BM25")

    pt.Experiment(
        [bm25],
        dataset.get_topics(),
        dataset.get_qrels(),
        eval_metrics=[RR @ 10, nDCG @ 20, MAP],
    )

if not pt.started():
    pt.init()
    
evaluation("vaswani")