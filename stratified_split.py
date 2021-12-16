from tqdm import tqdm
import numpy as np
import ijson


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
        split_1 = int(np.floor(0.96 * len(value)))
        split_2 = int(np.floor(0.98 * len(value)))
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
        if obj['pmid'] is not dev and obj['pmid'] is not test:
            train_set.append(obj)
        elif obj['pmid'] in dev:
            dev_set.append(obj)
        elif obj['pmid'] in test:
            test_set.append(obj)

    return train_set, dev_set, test_set






