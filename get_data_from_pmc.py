import argparse
import pickle

from tqdm import tqdm


def mapping_pmid2pmcid(filelist):
    id_pairs = []
    with open(filelist, 'r') as f:
        for line in f:
            info = line.split('\t')
            if len(info) <= 3:
                continue
            else:
                pmc_id = info[2]
                pmid = info[3]
                if pmid.startswith('PMID:'):
                    pmid = pmid[5:]
                id_pairs.append((pmc_id, pmid))
    return id_pairs


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--pmids')
    parser.add_argument('--save')
    args = parser.parse_args()

    id_pairs = mapping_pmid2pmcid(args.path)
    with open(args.pmids, "rb") as input_file:
        pmid = pickle.load(input_file)

    new_pairs = []
    for pair in tqdm(id_pairs):
        if pair[1] in pmid:
            new_pairs.append(pair)

    with open(args.save, "wb") as output_file:
        pickle.dump(new_pairs, output_file)


if __name__ == "__main__":
    main()