import random
import csv

missing_entry_prob = 0.9

data_miss_dist = [1, 0, 0]

full_data_file_name = 'studenbn1000_data.csv'

missing_data_file_name = '_'.join(
    [full_data_file_name.split('_', maxsplit=1)[0],
    str(missing_entry_prob)])
missing_data_file_name += '.csv'

data = []

with open(full_data_file_name, encoding='utf8') as full_data_file:
    fd_csv = csv.reader(full_data_file)
    for row in fd_csv:
        for i in range(len(row)):
            is_missing = random.random() <= missing_entry_prob
            if is_missing:
                row[i] = '-1'
        data.append(row)

with open(missing_data_file_name, 'w', encoding='utf8') as missing_data_file:
    md_csv = csv.writer(missing_data_file)
    md_csv.writerows(data)
