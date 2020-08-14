import pickle
import csv
from collections import Counter
from run_models import query_candidate


def read_mid_pkl(model_name, pkl_num):
    all_mid = []
    for i in range(pkl_num):
        with open('models_mid_result/第5轮的中间结果/{}{}.pkl'.format(model_name, str(i+1)), 'rb') as f:     # Path
            data = pickle.load(f)
            mid = data['model results']
            all_mid += mid
    print(len(all_mid))
    return all_mid


def count_votes():
    LDA_sim = read_mid_pkl('LDA', pkl_num=1)
    word2vec_sim = read_mid_pkl('word2vec', pkl_num=1)
    xlnet_sim = read_mid_pkl('xlnet', pkl_num=1)
    FT_sim = read_mid_pkl('FastText', pkl_num=1)
    bert_sim = read_mid_pkl('bert', pkl_num=1)
    LSTM_sim = read_mid_pkl('LSTM', pkl_num=1)

    assert len(LDA_sim) == len(word2vec_sim) == len(FT_sim) == len(xlnet_sim) == len(bert_sim) == len(LSTM_sim)

    query_result = query_candidate(start_num=1, end_num=117660, for_count_votes=True)
    eee = 0
    lda_er = word2vec_er = xlnet_er = FT_er = bert_er = LSTM_er = 0
    c = []
    empty_count = 0
    all_most_counter, model_score, all_m = [], [0, 0, 0, 0, 0, 0], []
    for i in range(len(LDA_sim)):
        if len(LDA_sim[i]) == 0:   # candidate set is empty
            empty_count += 1
            continue
        max1 = LDA_sim[i].index(max(LDA_sim[i]))
        max2 = word2vec_sim[i].index(max(word2vec_sim[i]))
        max3 = xlnet_sim[i].index(max(xlnet_sim[i]))
        max4 = FT_sim[i].index(max(FT_sim[i]))
        all_max = [max1, max2, max3, max4]
        max5 = bert_sim[i].index(max(bert_sim[i]))
        all_max.append(max5)
        max6 = LSTM_sim[i].index(max(LSTM_sim[i]))
        all_max.append(max6)

        wiki_des = [j[1] for j in query_result[i][1]]
        all_max_des = [wiki_des[j] for j in all_max]
        all_m.append(all_max_des)

        a = len(set(wiki_des))
        lda = len(set(LDA_sim[i]))
        if lda > a:
            lda_er += 1
        word2vec = len(set(word2vec_sim[i]))
        if word2vec > a:
            word2vec_er += 1
        xlnet = len(set(xlnet_sim[i]))
        if xlnet > a:
            xlnet_er += 1
        FT = len(set(FT_sim[i]))
        if FT > a:
            FT_er += 1
        bert = len(set(bert_sim[i]))
        if bert > a:
            bert_er += 1
        LSTM = len(set(LSTM_sim[i]))
        if LSTM > a:
            LSTM_er += 1

        counter = Counter(all_max_des)
        most_counter = counter.most_common(1)[0]
        c.append(most_counter[1])
        if len(set(wiki_des)) != 1:
            for key, value in counter.items():
                if value == 2:
                    for index, k in enumerate(all_max_des):
                        if k == key:
                            model_score[index] += 0.2
                if value == 3:
                    for index, k in enumerate(all_max_des):
                        if k == key:
                            model_score[index] += 0.4
                if value == 4:
                    for index, k in enumerate(all_max_des):
                        if k == key:
                            model_score[index] += 0.6
                if value == 5:
                    for index, k in enumerate(all_max_des):
                        if k == key:
                            model_score[index] += 0.8
                if value == 6:
                    for index, k in enumerate(all_max_des):
                        if k == key:
                            model_score[index] += 1
        else:
            eee += 1

        all_most_counter.append((most_counter[0], str(most_counter[1])+'|'+str(len(all_max))))
    counter = Counter(c)
    print(counter)
    print(lda_er, word2vec_er, xlnet_er, FT_er, bert_er, LSTM_er)
    print(eee)
    print(model_score)
    print('synset number:', len(LDA_sim))
    print('candidate empty number:{}'.format(empty_count))
    print('saved synset number:', len(all_most_counter))


if __name__ == "__main__":
    count_votes()
