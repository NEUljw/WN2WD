"""generate embeddings of descriptions"""
from sentence_transformers import SentenceTransformer
from datetime import datetime
import numpy as np
import csv
import pickle
from tqdm import tqdm


class Config:
    wordnet_path = 'merged_candidate.pkl'
    wiki_pkl_path = 'wiki_new.pkl'
    all_candidate_qnode_path = 'all_candidate_qnode.pkl'

    model_name = 'distilbert-base-nli-stsb-mean-tokens'    # Model name
    embed_save_path = 'distilbert_id2embed.pkl'    # Path to the result file (embeddings)
    batch_size = 128


def read_wordnet():
    with open(Config.wordnet_path, 'rb') as f:
        wn_data = pickle.load(f)
    print(len(wn_data))
    wn_des, wn_id = [], []
    for key, value in wn_data.items():
        wn_des.append(key[1])
        wn_id.append(key[0])
    return wn_des, wn_id


def read_wikidata():
    print('loading wikidata pkl file(dict)...')
    with open(Config.wiki_pkl_path, 'rb') as f:
        qnode2wiki = pickle.load(f)
    print('wikidata number:', len(qnode2wiki))

    with open(Config.all_candidate_qnode_path, 'rb') as f:
        all_qnodes = pickle.load(f)
    print('candidate qnode number:', len(all_qnodes))

    wiki_des, wiki_id = [], []
    for qnode in tqdm(all_qnodes, desc='loading candidate descriptions'):
        wiki_id.append(qnode)
        wiki_des.append(qnode2wiki[qnode][1])
    return wiki_des, wiki_id


def des_encode():
    start = datetime.now()
    print('start time:', start.strftime("%Y-%m-%d %H:%M:%S"), '\n==================')
    print('model name:', Config.model_name)


    wn_des, wn_id = read_wordnet()
    wiki_des, wiki_id = read_wikidata()
    des_lst = wn_des + wiki_des
    id_lst = wn_id + wiki_id

    print('des number:', len(des_lst))
    embedder = SentenceTransformer(Config.model_name)
    # handling 'None' in case of error
    des_lst = [i if i != 'None' else 'None Description' for i in des_lst]
    des_embeddings = embedder.encode(des_lst, batch_size=Config.batch_size, show_progress_bar=True)    # 768
    print(len(des_embeddings))

    id2embed = {}
    for id, embed in tqdm(zip(id_lst, des_embeddings), desc='creating embed dict'):
        id2embed[id] = embed

    print('saving sentence embeddings to file...')
    with open(Config.embed_save_path, 'wb') as f:
        pickle.dump(id2embed, f)
    print('save done！')

    end = datetime.now()
    print('==================\nend time:', end.strftime("%Y-%m-%d %H:%M:%S"))
    print('running time is', round((end-start).seconds/60, 4), 'minutes')


if __name__ == '__main__':
    des_encode()
