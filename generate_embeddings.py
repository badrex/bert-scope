# basic imports 
import sys
from collections import defaultdict, Counter
import numpy as np
import glob
import json
from tqdm import tqdm
import pickle

# import embedding framework FastEmbed
from fastembed import TextEmbedding
from typing import List, Dict


# parse command line arguments 
model_id = sys.argv[1]
output_file = sys.argv[2]

# read articles from desk
text_path = 'data/outputs/article_texts_full/' 

# get all files in the path 
text_files: List[str] = glob.glob(text_path + '*.txt')

documents: List[str] = []
document_IDs: List[str] = []

for text_path in text_files:
    with open(text_path, 'r') as f:
        document_IDs.append(text_path.split('/')[-1].split('.')[0])

        article = f.read()
        documents.append(article)


print(f"Number of articles: {len(documents)}")
 

# get supported models from TextEmbedding
supported_models = TextEmbedding.list_supported_models()

for m in supported_models:
    print(f"{m['model']:<60} {m['dim']:>5}")

# this will trigger the model download and initialization
#model_id = 'BAAI/bge-small-en-v1.5'
embedding_model = TextEmbedding(model_id)
print(f"The model {model_id} is ready to use.")

embeddings_generator = embedding_model.embed(documents)  

embeddings: List[str] = [] 

for emb in tqdm(embeddings_generator, total=len(documents)):
    embeddings.append(emb)


# turn embedding array into a dict to preserve article IDs
doc2embedding: Dict[str, np.array] = {
    doc: emb for doc, emb in zip(document_IDs, embeddings)
}

# save dict into disc as a pickle object
# output_file = doc2embedding.pkl'
with open(f'data/{output_file}', 'wb') as f:
    pickle.dump(doc2embedding, f)