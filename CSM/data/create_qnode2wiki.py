"""preprocess Wikidata data"""
import pickle
from tqdm import tqdm

wiki_pkl_path = 'datasets/wiki.pkl'     # Path to wiki.pkl
wiki_new_pkl_path = 'wiki_new.pkl'       # Path to the result file


# Load wiki.pkl
print('loading wikidata pkl file...')
with open(wiki_pkl_path, 'rb') as f:
    wiki_data = pickle.load(f)
print('wikidata number:', len(wiki_data))

qnode2wiki = {}
for one_wiki in tqdm(wiki_data, desc='creating dict'):
    qnode2wiki[one_wiki[0]] = one_wiki
print(len(qnode2wiki))

print('saving...')
with open(wiki_new_pkl_path, 'wb') as f:
    pickle.dump(qnode2wiki, f)
print('saving done!')
