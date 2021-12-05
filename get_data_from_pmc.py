import argparse
import json
import os
import pickle
import xml.etree.ElementTree as ET

from tqdm import tqdm

"""
steps to get pmc files:
1. use the pmid to pmc id mapping to obtain all pmc ids
2. all pmc XML files are splits by pmc ids, create the file name list according to PMC ids (https://ftp.ncbi.nlm.nih.gov/pub/wilbur/BioC-PMC/)
3. download each subset and use remove_pmc_file.sh to remove un-related files.
"""


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


def get_pmc_sublist(sublist):
    pmc = []
    for pair in sublist:
        pmc.append(pair[0])
    return pmc


def split_pmc_id(id_pairs):
    list00_30, list30_35, list35_40, list40_45, list45_50, list50_55, list55_60, list60_65, list65_70, list70_75, list75_80, list80 = ([] for i in range(12))
    for pair in id_pairs:
        pmc = int(pair[0][3:])
        if pmc < 3000000:
            list00_30.append(pair)
        elif 3000000 <= pmc < 3500000:
            list30_35.append(pair)
        elif 3500000 <= pmc < 4000000:
            list35_40.append(pair)
        elif 4000000 <= pmc < 4500000:
            list40_45.append(pair)
        elif 4500000 <= pmc < 5000000:
            list45_50.append(pair)
        elif 5000000 <= pmc < 5500000:
            list50_55.append(pair)
        elif 5500000 <= pmc < 6000000:
            list55_60.append(pair)
        elif 6000000 <= pmc < 6500000:
            list60_65.append(pair)
        elif 6500000 <= pmc < 7000000:
            list65_70.append(pair)
        elif 7000000 <= pmc < 7500000:
            list70_75.append(pair)
        elif 7500000 <= pmc < 8000000:
            list75_80.append(pair)
        else:
            list80.append(pair)

    pmc00_30 = get_pmc_sublist(list00_30)
    return pmc00_30


def pmc2pmid(file):
    mapping_id = {}
    with open(file, 'r') as f:
        for line in f:
            (key, value) = line.split(' ')
            mapping_id[key] = value.strip()
    return mapping_id


def get_data_from_pubmed_xml(file, mapping, pmc_id):

    mapping_id = pmc2pmid(mapping)

    tree = ET.parse(file)
    root = tree.getroot()
    document = root.find('document')

    pmc = 'PMC' + document.find('id').text
    try:
        pmid = mapping_id[pmc]
    except KeyError:
        pmid = mapping_id[pmc_id]
    data_point = {}
    intro = []
    methods = []
    results = []
    discuss = []
    figure_captions = []
    table_captions = []
    for passage in document.findall('passage'):
        for infon in passage.findall('infon'):
            if infon.attrib['key'] == 'section_type':
                if infon.text == 'INTRO':
                    if passage.find('text').text.endswith('.'):
                        intro.append(passage.find('text').text)
                    else:
                        intro.append(passage.find('text').text + '.')
                elif infon.text == 'METHODS':
                    if passage.find('text').text.endswith('.'):
                        methods.append(passage.find('text').text)
                    else:
                        methods.append(passage.find('text').text + '.')
                elif infon.text == 'RESULTS':
                    if passage.find('text').text.endswith('.'):
                        results.append(passage.find('text').text)
                    else:
                        results.append(passage.find('text').text + '.')
                elif infon.text == 'DISCUSS':
                    if passage.find('text').text.endswith('.'):
                        discuss.append(passage.find('text').text)
                    else:
                        discuss.append(passage.find('text').text + '.')
            elif infon.attrib['key'] == 'type':
                if infon.text == 'fig_title_caption' or infon.text == 'fig_caption':
                    figure_captions.append(passage.find('text').text)
                if infon.text == 'table_title_caption' or infon.text == 'table_footnote':
                    table_captions.append(passage.find('text').text)
                continue
    data_point['pmid'] = pmid
    data_point['INTRO'] = ' '.join(intro)
    data_point['METHODS'] = ' '.join(methods)
    data_point['RESULTS'] = ' '.join(results)
    data_point['DISCUSS'] = ' '.join(discuss)
    data_point['FIG_CAPTIONS'] = ' '.join(figure_captions)
    data_point['TABLE_CAPTIONS'] = ' '.join(table_captions)

    return data_point


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--pmids')
    parser.add_argument('--id_mapping')
    parser.add_argument('--save')
    args = parser.parse_args()

    # get id pairs
    # id_pairs = mapping_pmid2pmcid(args.path)
    # with open(args.pmids, "rb") as input_file:
    #     pmid = pickle.load(input_file)
    #
    # new_pairs = []
    # for pair in tqdm(id_pairs):
    #     if pair[1] in pmid:
    #         new_pairs.append(pair)
    #
    # with open(args.save, "wb") as output_file:
    #     pickle.dump(new_pairs, output_file)

    # get pmc full text
    dataset = []
    for root, dirs, files in os.walk(args.path):
        for file in tqdm(files):
            filename, extension = os.path.splitext(file)
            if extension == '.xml':
                pmc_backup = filename[3:].strip()
                data_point = get_data_from_pubmed_xml(file, args.id_mapping, pmc_backup)
                dataset.append(data_point)

    pubmed = {'articles': dataset}
    with open(args.save, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)


if __name__ == "__main__":
    main()