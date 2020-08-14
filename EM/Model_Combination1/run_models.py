import time
import threading
import pickle

from models.LDA.LDA_model import cal_sim_LDA
from models.word2vec.word2vec_model import cal_sim_word2vec
from models.Xlnet.fine_xlnet import cal_sim_fine_xlnet
from models.BERT.fine_bert_model import cal_sim_fine_bert
from models.FastText.FastText_use import cal_sim_FastText
from models.MaLSTM.model_predict import data_to_csv, cal_sim_lstm


def query_candidate(des_none_sim=-100, run_models=None, start_num=0, end_num=0,
                    for_count_votes=False, gpu_name=None, gpu_num=None, batch_size=None):
    with open('data/run_data.pkl', 'rb') as f:        # Path
        data = pickle.load(f)
        query_result = data['data']
    query_result = query_result[start_num-1:end_num-1]
    start_num = str(start_num)
    end_num = str(end_num)

    if for_count_votes is True:
        return query_result

    all_synset_des, all_wiki_candidate, all_synset_id = [], [], []
    for i, value in enumerate(query_result):
        synset_des = value[0][1]
        synset_id = value[0][0]
        all_synset_des.append(synset_des)
        all_synset_id.append(synset_id)
        wiki_candidate = [w[1] for w in value[1]]
        all_wiki_candidate.append(wiki_candidate)
    print('create data end!')

    # calculate mid results
    print('wordnet synsets number:', len(all_synset_des))
    if 'LDA' in run_models:
        print('LDA model running..')
        LDA_sim = cal_sim_LDA(all_synset_des, all_wiki_candidate, default_sim=des_none_sim)
        with open('models_mid_result/LDA_{}_{}.pkl'.format(start_num, end_num), 'wb') as f:
            pickle.dump({'model results': LDA_sim}, f)

    if 'word2vec' in run_models:
        print('word2vec model running..')
        word2vec_sim = cal_sim_word2vec(all_synset_des, all_wiki_candidate, default_sim=des_none_sim)
        with open('models_mid_result/word2vec_{}_{}.pkl'.format(start_num, end_num), 'wb') as f:
            pickle.dump({'model results': word2vec_sim}, f)

    if 'xlnet' in run_models:
        print('xlnet model running..')
        xlnet_sim = cal_sim_fine_xlnet(all_synset_des, all_wiki_candidate, gpu_name, gpu_num, batch_size)
        with open('models_mid_result/xlnet_{}_{}.pkl'.format(start_num, end_num), 'wb') as f:
            pickle.dump({'model results': xlnet_sim}, f)

    if 'bert' in run_models:
        print('bert model running..')
        bert_sim = cal_sim_fine_bert(all_synset_des, all_wiki_candidate, gpu_name, gpu_num, batch_size)
        with open('models_mid_result/bert_{}_{}.pkl'.format(start_num, end_num), 'wb') as f:
            pickle.dump({'model results': bert_sim}, f)

    if 'FastText' in run_models:
        print('FastText model running..')
        FT_sim = cal_sim_FastText(all_synset_des, all_wiki_candidate)
        with open('models_mid_result/FastText_{}_{}.pkl'.format(start_num, end_num), 'wb') as f:
            pickle.dump({'model results': FT_sim}, f)

    if 'LSTM' in run_models:
        print('LSTM model running..')
        print('format conversion...')
        data_to_csv(all_synset_des, all_wiki_candidate)
        wiki_len = [len(q) for q in all_wiki_candidate]
        LSTM_sim = cal_sim_lstm(wiki_len=wiki_len, gpu_name=gpu_name, gpu_num=gpu_num)
        with open('models_mid_result/LSTM_{}_{}.pkl'.format(start_num, end_num), 'wb') as f:
            pickle.dump({'model results': LSTM_sim}, f)

    print('--'*20+' end '+'--'*20)


def run(syn_start, syn_end, run_models, gpu_name=None, gpu_num=None, batch_size=24):

    print('running models：', run_models)
    print('--'*20+' start '+'--'*20)
    start = time.clock()

    des_none_sim = -100
    print('create data start...')
    query_candidate(des_none_sim=des_none_sim,
                    run_models=run_models,
                    start_num=syn_start,
                    end_num=syn_end,
                    gpu_name=gpu_name,
                    gpu_num=gpu_num,
                    batch_size=batch_size
                    )

    end = time.clock()
    print('use time:', (end-start)/60, ' minutes')


if __name__ == "__main__":
    '''
    1. Models include: 'LDA'、'word2vec'、'xlnet'、'bert'、'FastText'、'LSTM'
    2. Support multi GPUs
    3. BERT and XLNet shouldn't be in the same thread
    '''
    gpu_name = "0"      # GPU id
    gpu_num = 1       # GPU number
    step = 2          # running step
    if step == 1:
        t1 = threading.Thread(target=run, args=(1, 117660, ['LDA', 'word2vec'],))
        t2 = threading.Thread(target=run, args=(1, 117660, ['FastText'],))
        t1.start()
        t2.start()
    if step == 2:
        run(1, 117660, run_models=['LSTM'], gpu_name=gpu_name, gpu_num=gpu_num)     # except first round
        run(1, 117660, run_models=['xlnet'], gpu_name=gpu_name, gpu_num=gpu_num, batch_size=12)
        run(1, 117660, run_models=['bert'], gpu_name=gpu_name, gpu_num=gpu_num, batch_size=12)
