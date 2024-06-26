import pyterrier as pt
import os

# This script is used to create the indexes from the cord19 fulltext corpus
pt.init()
dataset = pt.datasets.get_dataset('irds:cord19/fulltext')
indexer = pt.IterDictIndexer("./indices/cord19_fulltext")
index_ref = indexer.index(dataset.get_corpus_iter(), fields=['title', 'doi', 'date', 'abstract'])