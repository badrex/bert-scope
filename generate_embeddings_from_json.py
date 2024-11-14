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

# read articles from from yaml files
text_path = 'data/json_files/' 

# get all files in the path 
json_files: List[str] = glob.glob(text_path + '*.json')

print(json_files[0])

doc_summaries: List[str] = []
doc_IDs: List[str] = []

# read all yaml files
for file in json_files:
    with open(file, 'r') as f:
        article_json = json.load(f)

        doc_summary = '\n\n'.join(
            [article_json['revised_title'], article_json['tldr']]
        )

        doc_summaries.append(doc_summary)
        doc_IDs.append(file.split('/')[-1].split('.')[0])

# print(doc_summaries[0]) 
# print(doc_IDs[0]) 

print(f"Number of articles: {len(doc_summaries)}")
 

#get supported models from TextEmbedding
supported_models = TextEmbedding.list_supported_models()

for m in supported_models:
    print(f"{m['model']:<60} {m['dim']:>5}")

# this will trigger the model download and initialization
#model_id = 'BAAI/bge-small-en-v1.5'
embedding_model = TextEmbedding(model_id)
print(f"The model {model_id} is ready to use.")

# generate embeddings for all articles
embeddings_generator = embedding_model.embed(doc_summaries)  

embeddings: List[str] = [] 

for emb in tqdm(embeddings_generator, total=len(doc_summaries)):
    embeddings.append(emb)


# turn embedding array into a dict to preserve article IDs
doc2embedding: Dict[str, np.array] = {
    doc: emb for doc, emb in zip(doc_IDs, embeddings)
}

# save dict into disc as a pickle object
# output_file = doc2embedding.pkl'
with open(f'data/{output_file}', 'wb') as f:
    pickle.dump(doc2embedding, f)