import argparse
from tqdm import tqdm
import numpy as np
import ijson
import json
import pandas as pd


def json2csv(in_file, out_file):
    f = open(in_file, encoding="utf8")
    objects = ijson.items(f, 'articles.item')
    articles = []
    mesh = []
    for i, obj in enumerate(tqdm(objects)):
        doc = {}
        doc["pmid"] = obj["pmid"]
        title = obj['title'].strip()
        abstract = obj['abstractText'].strip()
        doc["title_abstract"] = title + ' ' + abstract
        doc['INTRO'] = obj['INTRO']
        doc['METHODS'] = obj['METHODS']
        doc['RESULTS'] = obj['RESULTS']
        doc['DISCUSS'] = obj['DISCUSS']
        mesh_id = list(obj['mesh'].keys())
        articles.append(doc)
        mesh.append(mesh_id)

    df = pd.DataFrame(articles)
    df['mesh'] = pd.Series(mesh)
    df.to_csv(out_file, index=False)


def stratified_split(path):
    f = open(path, encoding="utf8")
    objects = ijson.items(f, 'articles.item')

    year = {}
    for i, obj in enumerate(tqdm(objects)):
        if obj['year'] in year:
            year[obj['year']].append(obj['pmid'])
        else:
            year[obj['year']] = []
            year[obj['year']].append(obj['pmid'])

    split = {}
    for key, value in year.items():
        split[key] = {}
        split_1 = int(np.floor(0.8 * len(value)))
        split_2 = int(np.floor(0.9 * len(value)))
        split[key]['train'] = value[:split_1]
        split[key]['dev'] = value[split_1:split_2]
        split[key]['test'] = value[split_2:]

    train = []
    dev = []
    test = []
    for key, value in split.items():
        train.extend(value['train'])
        dev.extend(value['dev'])
        test.extend(value['test'])

    train_set = []
    dev_set = []
    test_set = []
    for i, obj in enumerate(tqdm(objects)):
        if obj['pmid'] not in dev and obj['pmid'] not in test:
            train_set.append(obj)
        elif obj['pmid'] in dev:
            dev_set.append(obj)
        elif obj['pmid'] in test:
            test_set.append(obj)

    return train_set, dev_set, test_set


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--save_train_path')
    parser.add_argument('--save_dev_path')
    parser.add_argument('--save_test_path')

    args = parser.parse_args()

    train_set, dev_set, test_set = stratified_split(args.path)

    pubmed_train = {'articles': train_set}
    pubmed_dev = {'articles': dev_set}
    pubmed_test = {'articles': test_set}

    with open(args.save_train_path, "w") as outfile:
        json.dump(pubmed_train, outfile, indent=4)

    with open(args.save_dev_path, "w") as outfile:
        json.dump(pubmed_dev, outfile, indent=4)

    with open(args.save_test_path, "w") as outfile:
        json.dump(pubmed_test, outfile, indent=4)


if __name__ == "__main__":
    main()




