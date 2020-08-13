# Mapping of WordNet to Wikidata



## Introduction

We propose an **Ensemble Method (EM)** and a **Comprehensive Similarity Method (CSM)** to map WordNet to Wikidata. Among them, CSM has the highest correct rate, the final mapping results is `mapping results.pkl`.



## Setup

We recommend Python 3.6 or higher. The models is implemented with Tensorflow (1.13.1) and Pytorch (at least 1.0.1).

The code does **not** work with Python 2.7.



## Data preprocess

First of all, we need to preprocess the data.

1. **Download the relevant data and put it in the corresponding path.**
2. Set the path parameters in `create_data.py` and run it to preprocess the original data.
3. Set the path parameters in `create_run_data.py` and run to generate the running data.

Then, we can run any of the three methods:

+ EM (model combination1)
+ EM (model combination2)
+ CSM 



## Ensemble Method (EM)

For the model selection, we use two combinations.

### Model Combination1

There are six models in this model combination, including three Non-Contextual models: LDA, Word2Vec, FastText, and three Contextual models: MaLSTM, BERT, XLNet.

Run multiple iterations, the steps for one iteration are as follows:

1. Set the path parameters in `models/LDA/LDA_model.py`, run the corresponding part, generate the train data of None-Contextual models and train LDA model.
2. Set the path parameters in `models/word2vec/word2vec_model.py` and run it to train Word2Vec model.
3. Set the path parameters in `models/FastText/FastText_train.py` and run it to train FastText model.
4. Set the parameters in `models/BERT/bert_train.py` and run to fine-tune the BERT model (except the first round).
5. Set parameters in `models/Xlnet/xlnet_train.py` and run to fine-tune the XLNet model (except the first round).
6. Set parameters in `models/MaLSTM/MaLSTM (the second round) or MaLSTM2 (after the second round)` and run to train MaLSTM model.
7. Set the parameters in `run_models.py` and run the corresponding part to generate intermediate results.
8. Set the parameters in `cal_score.py` and run it to calculate the model score (except the first round).
9. Set the parameters in `count_votes.py` and run the corresponding part, generate mapping results and negative train data of the Contextual models according to the intermediate results.
10. Set the parameters in `create_train_data.py` and run it to generate the positive train data of Contextual models and merge it with negative train data.
11. Set the parameters in `merge_train_data.py` and run the corresponding part, merge all the previous corresponding train data as the train data of Contextual models.

### Model Combination2

There are eight models in this model combination, including three Non-Contextual models: LDA, Word2Vec, FastText, and five Contextual models: Siamese LSTM, Siamese XLNet, Siamese BERT, Siamese RoBERTa, Siamese DistilBERT.

First, generate the intermediate results of None-Contextual models.

1. Set the parameters in `create_train_data_ML.py` and run it to generate the train data of None-Contextual models.
2. Set the parameters in `models/LDA/LDA_model.py` and run to train LDA model.
3. Set the parameters in `models/word2vec/word2vec_model.py` and run to train Word2Vec model.
4. Set the parameters in `models/FastText/FastText_train.py` and run to train FastText model.
5. Set the parameters in `run_models.py` and run to generate intermediate results.

Second, generate the intermediate results of Contextual models.

1. Set parameters and run `training/xlnet_train_on_nli.py` and `xlnet_continue_train_on_sts.py` successively to fine-tune XLNet.
2. Set parameters and run `lstm_train_on_sts.py` to train LSTM.
3. Set parameters and run `run_models.py` to generate intermediate results.

Finally, all intermediate results are processed to generate mapping results.

1. Set the parameters and run `cal_mid_score.py` to process the intermediate results and generate the mapping results.



## Comprehensive Similarity Method (CSM) 

We combine the **description similarity** and the **label similarity** as the comprehensive similarity.

The operation steps are as follows:

1. Download data from WikiData official website, a total of 10000 parts (needs a lot of disk space).
2. **Download the rest of the relevant data and put it in the corresponding path.**
3. Set the path parameters in `data/parse_wikidata.py` and run it to parse the original Wikidata data.
4. Set the path parameters in `data/create_qnode2wiki.py` and run it to process the data format.
6. Set the path parameters in `create_wiki_dict.py` and run it to generate a dictionary.
7. Set the path parameters in `create_dict_from_web_results.py` and run it to generate another dictionary.
8. Set the path parameters in `create_candidate_lab.py` and run to generate the first candidate set.
9. Set the path parameters in `create_candidate_lab.py` and run to generate the second candidate set.
10. Set the path parameters in `merge_candidate.py` and run to merge the two candidate sets.
11. Set the path parameters in `EM/Model_Combination2/Contextual_Models/encode_and_save.py` and run to generate the embeddings of descriptions.
11. Set the path parameters in `embed_lab_sim.py` and run it to generate the final mapping results.