"""generate mapping results"""
import pickle
from tqdm import tqdm
import numpy as np


class Config:
    wordnet_path = 'merged_candidate.pkl'
    wiki_path = 'wiki_new.pkl'
    embed_path = 'distilbert_id2embed.pkl'

    result_path = 'distilbert_result.pkl'      # Path to the result file (mapping results)

    lab_add = 100
    lab_add_more = 100
    none_sim = 0
    lab_num_limit = 3


def cosine_sim(a, b):
    try:
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    except ValueError:
        sim = 0
    return round(sim, 4)


def read_wordnet():
    with open(Config.wordnet_path, 'rb') as f:
        wn_data = pickle.load(f)
    print('synset number:', len(wn_data))
    return wn_data


def cal():
    wn2result = {}
    wn = read_wordnet()

    print('loading qnode2wiki...')
    with open(Config.wiki_path, 'rb') as f:
        qnode2wiki = pickle.load(f)
    print('qnode2wiki length:', len(qnode2wiki))

    print('loading id2embed...')
    with open(Config.embed_path, 'rb') as f:
        id2embed = pickle.load(f)
    print('id2embed length:', len(id2embed))

    for one_wn, candi in tqdm(wn.items(), desc='calculating'):
        one_wn_embed = id2embed[one_wn[0]]
        one_wn_labs = one_wn[2]
        one_wn_sim, one_wn_des_sim, one_wn_lab_sim = [], [], []
        for one_candi in candi:
            one_candi_wiki = qnode2wiki[one_candi]
            one_candi_des = one_candi_wiki[1]
            one_candi_labs = one_candi_wiki[2]
            one_candi_embed = id2embed[one_candi]
            # des sim
            if one_candi_des != 'None':
                des_sim = cosine_sim(one_wn_embed, one_candi_embed)
            else:
                des_sim = Config.none_sim
            one_wn_des_sim.append(des_sim)
            # lab sim
            lab_inter_num = len(set(one_wn_labs).intersection(set(one_candi_labs)))
            if lab_inter_num > 0:
                if len(one_candi_labs) > Config.lab_num_limit:
                    lab_sim = Config.lab_add * lab_inter_num
                else:
                    lab_sim = Config.lab_add_more * lab_inter_num
            else:
                lab_sim = 0
            one_wn_lab_sim.append(lab_sim)
            # sim
            sim = des_sim + lab_sim
            one_wn_sim.append(sim)
        max_sim = max(one_wn_sim)
        max_sim_idx = one_wn_sim.index(max_sim)
        one_wn_lab_sim_sorted = sorted(one_wn_lab_sim, reverse=True)
        one_wn_des_sim_sorted = sorted(one_wn_des_sim, reverse=True)
        wn2result[one_wn] = qnode2wiki[candi[max_sim_idx]] + [max_sim,
                                                              one_wn_lab_sim[max_sim_idx],
                                                              one_wn_des_sim[max_sim_idx],
                                                              len(candi),
                                                              one_wn_lab_sim_sorted.index(one_wn_lab_sim[max_sim_idx])+1,
                                                              one_wn_des_sim_sorted.index(one_wn_des_sim[max_sim_idx])+1]

    print('saving mapping results...')
    with open(Config.result_path, 'wb') as f:
        pickle.dump(wn2result, f)
    print('done!')


if __name__ == '__main__':
    cal()
