import pyterrier as pt
import os

pt.init()
dataset = pt.datasets.get_dataset('irds:cord19/fulltext')
indexer = pt.IterDictIndexer("./indices/cord19_fulltext")
index_ref = indexer.index(dataset.get_corpus_iter(), fields=['title', 'doi', 'date', 'abstract'])