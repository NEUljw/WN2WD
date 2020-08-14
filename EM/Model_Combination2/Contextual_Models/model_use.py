from sentence_transformers import SentenceTransformer
import scipy.spatial
from datetime import datetime


def cal_sim(wn_des_list, wiki_des_list, model_path, batch_size):
    start = datetime.now()
    print('start time:', start.strftime("%Y-%m-%d %H:%M:%S"), '\n==================')
    print('model path:', model_path)
    print('synset number:', len(wn_des_list))
    embedder = SentenceTransformer(model_path)
    wiki_len = [len(i) for i in wiki_des_list]
    wiki_des_list = sum(wiki_des_list, [])
    # handling 'None' to prevent error
    wiki_des_list = [i if i != 'None' else 'None Description' for i in wiki_des_list]

    wiki_embeddings = embedder.encode(wiki_des_list, batch_size=batch_size)
    wn_embeddings = embedder.encode(wn_des_list, batch_size=batch_size)
    print(len(wiki_embeddings), len(wn_embeddings))

    wiki_embeddings_lst = []
    for index, i in enumerate(wiki_len):
        wiki_embeddings_lst.append(wiki_embeddings[sum(wiki_len[:index]):sum(wiki_len[:index+1])])
    assert len(wiki_embeddings_lst) == len(wn_embeddings)
    print(len(wiki_embeddings_lst), len(wn_embeddings))

    all_sim_lst = []
    for wn_emb, wiki_emb in zip(wn_embeddings, wiki_embeddings_lst):
        sim_lst = []
        distances = scipy.spatial.distance.cdist([wn_emb], wiki_emb, "cosine")[0]
        results = zip(range(len(distances)), distances)

        for idx, distance in results:
            sim = round(1 - distance, 6)
            sim_lst.append(sim)
        all_sim_lst.append(sim_lst)

    end = datetime.now()
    print('==================\nend time:', end.strftime("%Y-%m-%d %H:%M:%S"))
    print('running time is', round((end-start).seconds/60, 4), 'minutes')
    return all_sim_lst
