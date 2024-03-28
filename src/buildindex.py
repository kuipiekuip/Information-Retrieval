import pyterrier as pt
import os

pt.init()
# list of filenames to index
file_path = os.path.expanduser('~') + "/.ir_datasets/nyt/"
files = pt.io.find_files(file_path)

# build the index
indexer = pt.TRECCollectionIndexer("./nyt_index", verbose=True, blocks=False)
indexref = indexer.index(files)

# load the index, print the statistics
index = pt.IndexFactory.of(indexref)
print(index.getCollectionStatistics().toString())