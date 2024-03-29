import pyterrier as pt
import os

pt.init()
dataset = pt.datasets.get_dataset('irds:nyt')
indexer = pt.IterDictIndexer("./indices/nyt")
index_ref = indexer.index(dataset.get_corpus_iter(), fields=['headline', 'body'])