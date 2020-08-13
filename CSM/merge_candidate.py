"""merge two candidate sets"""
import pickle
from tqdm import tqdm
from collections import Counter

with open('candidate_lab_web.pkl', 'rb') as f:     # Path
    web_candi = pickle.load(f)
with open('candidate_lab.pkl', 'rb') as f:      # Path
    lab_candi = pickle.load(f)

print(len(web_candi), len(lab_candi))

merge_candi = {}
for key in tqdm(web_candi):
    if len(set(web_candi[key]+lab_candi[key])) > 0:
        merge_candi[key] = list(set(web_candi[key]+lab_candi[key]))

with open('merged_candidate.pkl', 'wb') as f:     # Path
    pickle.dump(merge_candi, f)
print('done!')




with open('merged_candidate.pkl', 'rb') as f:    # Path
    m_candi = pickle.load(f)
print(len(m_candi))


all_candi_qnodes = []
for key, value in m_candi.items():
    all_candi_qnodes += value
print(len(all_candi_qnodes))
all_candi_qnodes = list(set(all_candi_qnodes))
print(len(all_candi_qnodes))

print('saving qnodes...')
with open('all_candidate_qnode.pkl', 'wb') as f:   # Path
    pickle.dump(all_candi_qnodes, f)
print('saving qnodes done')
