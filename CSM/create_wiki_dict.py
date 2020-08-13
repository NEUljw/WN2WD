"""generate a dict"""
import pickle
from nltk.stem.porter import PorterStemmer
from tqdm import tqdm


class Config:
    wiki_pkl_path = 'datasets/wiki.pkl'   # Path to wiki.pkl
    dict_path = 'word2qnodes.pkl'    # Path to the result file (dict)


def read_wikidata():
    # Load wiki.pkl
    print('loading wikidata pkl file...')
    with open(Config.wiki_pkl_path, 'rb') as f:
        wiki_data = pickle.load(f)
    print('wikidata number:', len(wiki_data))
    return wiki_data


def create_dict(wiki):
    word2qnodes = {}
    porter_stemmer = PorterStemmer()

    for one_wiki in tqdm(wiki, desc='creating word2qnodes dict'):
        if one_wiki[0][0] == 'Q' and one_wiki[2] != ['None']:
            labels = list(set(one_wiki[2]))
            labels = [i if i.find(' ') >= 0 else porter_stemmer.stem(i) for i in labels]
            labels = list(set(labels))
            for one_label in labels:
                if one_label not in word2qnodes.keys():
                    word2qnodes[one_label] = [one_wiki[0]]
                else:
                    word2qnodes[one_label].append(one_wiki[0])
    print('dict length:', len(word2qnodes))

    with open(Config.dict_path, 'wb') as f:
        pickle.dump(word2qnodes, f)


if __name__ == '__main__':
    wiki = read_wikidata()
    create_dict(wiki)
