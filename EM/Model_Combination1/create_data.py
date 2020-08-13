"""preprocess the original data"""
import csv

csv.field_size_limit(500 * 1024 * 1024)


def create_wordnet_data():
    data = []
    with open('original_data_all/WN2WD_Mapping.csv', 'r', encoding='utf-8') as f:    # Path
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append([row[0], row[5], row[6]])

    with open('data/wordnet_data.csv', 'w', encoding='utf-8', newline='') as f:    # Path
        f_csv = csv.writer(f)
        f_csv.writerows(data)


def list_to_str(data, replace_blank=True):
    def replace_blanks(word):
        # blank to _
        after_word = word.replace(' ', '_')
        return after_word

    data_str = ''
    if replace_blank is True:
        for i in data:
            data_str = data_str+replace_blanks(i)+','
    else:
        for i in data:
            data_str = data_str+i+','
    data_str = data_str[:-1]
    return data_str


def structure_candidate(file_path, new_file_path, top_n):
    all_count, empty_count, error_count, right_count = 0, 0, 0, 0
    candidate_dict = dict()
    with open(file_path, encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            all_count += 1
            row_results = []
            query_word = row[0]
            row_dict = eval(row[1])

            # candidate set is empty
            try:
                if row_dict['hits']['total'] == 0:
                    candidate_dict[query_word] = []
                    empty_count += 1
                    continue
            # query result error
            except KeyError:
                candidate_dict[query_word] = []
                error_count += 1
                continue

            for one_r in row_dict['hits']['hits']:
                one_source = one_r['_source']
                one_qnode = one_source['title']
                if 'descriptions' not in one_source.keys():
                    one_desc = 'None'
                else:
                    if 'en' not in one_source['descriptions'].keys():
                        one_desc = 'None'
                    else:
                        one_desc = one_source['descriptions']['en'][0]

                if 'labels' not in one_source.keys():
                    one_labels = 'None'
                else:
                    if 'en' not in one_source['labels'].keys():
                        one_labels = 'None'
                    else:
                        one_labels = one_source['labels']['en']
                        if len(one_labels) == 0:
                            one_labels = 'None'
                        else:
                            one_labels = list_to_str(one_labels, replace_blank=True)
                row_results.append([one_qnode, one_desc, one_labels])
            # top_n
            if len(row_results) > top_n:
                row_results = row_results[:top_n]
            candidate_dict[query_word] = row_results
            right_count += 1
    data = []
    for key, value in candidate_dict.items():
        all_value = [key]
        for k in value:
            all_value += k
        data.append(all_value)

    with open(new_file_path, 'w', encoding='utf-8-sig', newline='') as f:
        f_csv = csv.writer(f)
        f_csv.writerows(data)
    print('all count:{} | empty count:{} | error count:{} | right count:{}'.format(
        all_count, empty_count, error_count, right_count))


def create_candidate_wikidata(top_n):
    for file_num in range(1, 14):
        structure_candidate(file_path='original_data_all/top50/all_synsets_'+str(file_num)+'.csv',
                            new_file_path='data/wikidata_candidate/candidate_part_'+str(file_num)+'.csv',
                            top_n=top_n)         # Path
        print('number '+str(file_num)+' file done.')


if __name__ == '__main__':
    create_wordnet_data()
    create_candidate_wikidata(top_n=15)
