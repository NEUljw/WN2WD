import csv


def read_data(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        f_csv = csv.reader(f)
        head_row = next(f_csv)  # 跳过表头
        for row in f_csv:
            data.append(row)
    return data


a = read_data('train_all.csv')
b = read_data('train.csv')
all_data = a + b
print(len(a))
print(len(b))
print(len(all_data))

with open('train_new.csv', 'w', encoding='utf-8-sig', newline='') as f:
    f_csv = csv.writer(f)
    f_csv.writerow(['text1', 'text2', 'is_same'])
    f_csv.writerows(all_data)
