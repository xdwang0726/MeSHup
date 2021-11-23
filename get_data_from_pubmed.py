import argparse
import ijson
import json
import os
import pickle
import urllib.request
import xml.etree.ElementTree as ET

from bs4 import BeautifulSoup
from tqdm import tqdm


# get PMID from PMC
def get_pmids_from_pmc(filelist):

    """read file list from PMC at ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.txt"""

    pmids = []
    with open(filelist, 'r') as f:
        for line in f:
            info = line.split('\t')
            if len(info) <= 3:
                continue
            else:
                pmid = info[3]
                if pmid.startswith('PMID:'):
                    pmid = pmid[5:]
                pmids.append(pmid)
    pmids = list(set(list(filter(None, pmids))))

    return pmids


def check_if_document_is_mannually_curated(file):
    tree = ET.parse(file)
    root = tree.getroot()
    pmids = []
    for articles in root.findall('PubmedArticle'):
        medlines = articles.find('MedlineCitation')
        if 'IndexingMethod' in medlines.attrib:
            pmid = medlines.find('PMID').text
            # file_name = Path(file).name.strip('.xml')[6:]
            # pmid = file_name[:2] + str(version) + file_name[3:]
            pmids.append(pmid)
        else:
            continue
    pmids = list(set(pmids))
    return pmids


def get_data_from_pubmed_xml(file, pmc_list):

    tree = ET.parse(file)
    root = tree.getroot()

    dataset = []
    for articles in root.findall('PubmedArticle'):
        data_point = {}
        mesh_ids = []
        mesh_major = []
        medlines = articles.find('MedlineCitation')
        pmid = medlines.find('PMID').text
        article_info = medlines.find('Article')
        if 'IndexingMethod' in medlines.attrib or  medlines.find('MeshHeadingList') is None:
            continue
        elif article_info.find('ArticleTitle') is None or article_info.find('Abstract') is None:
            continue
        elif medlines.find('MeshHeadingList') is None:
            continue
        else:
            title = "".join(article_info.find('ArticleTitle').itertext())
            if title == 'Not Available' or title == 'In process':
                continue
            elif article_info.find('Abstract').find('AbstractText') is None:
                continue
            elif pmid in set(pmc_list):
                journal_info = article_info.find('Journal')
                year = journal_info.find('JournalIssue').find('PubDate')
                if year.find('Year') is None:
                    year = year.find('MedlineDate').text[:4]
                else:
                    year = year.find('Year').text
                journal_name = journal_info.find('Title').text
                abstract = []
                for ab in article_info.find('Abstract').findall('AbstractText'):
                    abstract.append("".join(ab.itertext()))
                abstract = list(filter(None, abstract))
                abstract = ' '.join(abstract)
                for mesh in medlines.find('MeshHeadingList').findall('MeshHeading'):
                    m = mesh.find('DescriptorName').attrib['UI']
                    m_name = mesh.find('DescriptorName').text
                    mesh_ids.append(m)
                    mesh_major.append(m_name)
                data_point['pmid'] = pmid
                data_point['title'] = title
                data_point['abstractText'] = abstract
                data_point["meshMajor"] = mesh_major
                data_point["meshID"] = mesh_ids
                data_point['journal'] = journal_name
                data_point['year'] = year
                dataset.append(data_point)

    return dataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--pmids')
    # parser.add_argument('--save')
    # parser.add_argument('--save_no_mesh')
    # parser.add_argument('--pmid_path')
    # parser.add_argument('--mapping_path')
    # parser.add_argument('--allMesh')
    parser.add_argument('--save_dataset')
    # parser.add_argument('--save_missed')

    args = parser.parse_args()

    pmcs_list = []
    with open(args.pmids, 'r') as f:
        for ids in f:
            pmcs_list.append(ids.strip())
    print('mannually annoted articles: %d' % len(pmcs_list))

    data = []
    for root, dirs, files in os.walk(args.path):
        for file in tqdm(files):
            filename, extension = os.path.splitext(file)
            if extension == '.xml':
                dataset = get_data_from_pubmed_xml(file, pmcs_list)
                data.extend(dataset)
    print('Total number of articles %d' % len(data))
    pubmed = {'articles': data}
    # no_mesh_pmid_list = list(set([ids for pmids in no_mesh for ids in pmids]))
    #
    # new_pmids = list(set(pmids_list) - set(no_mesh_pmid_list))
    # print('Total number of articles %d' % len(new_pmids))
    #
    # pickle.dump(no_mesh_pmid_list, open(args.save_no_mesh, 'wb'))
    # #
    # with open(args.save, 'w') as f:
    #     for ids in new_pmids:
    #         f.write('%s\n' % ids)

    # pubmed, missed_ids = get_data(args.pmid_path, args.mapping_path, args.allMesh)
    #
    # pubmed = merge_json(args.path)
    with open(args.save_dataset, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)

    #pickle.dump(missed_ids, open(args.save_missed, 'wb'))


if __name__ == "__main__":
    main()
