import json
import csv


def json2csv(in_file, out_file):
    with open(in_file) as json_file:
        data = json.load(json_file)

    article_data = data['articles']

    data_file = open(out_file, 'w')
    csv_writer = csv.writer(data_file)

    count = 0
    for article in article_data:
        if count == 0:
            # Writing headers of CSV file
            header = article.keys()
            csv_writer.writerow(header)
            count += 1
        # Writing data of CSV file
        csv_writer.writerow(article.values())

    data_file.close()
