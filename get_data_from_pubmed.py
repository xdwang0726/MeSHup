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
        mesh_list = {}
        author_list = []
        chemicals = {}
        doi = 'None'
        supply_mesh = {}
        medlines = articles.find('MedlineCitation')
        pmid = medlines.find('PMID').text
        article_info = medlines.find('Article')
        if 'IndexingMethod' in medlines.attrib or medlines.find('MeshHeadingList') is None:
            continue
        elif article_info.find('ArticleTitle') is None or article_info.find('Abstract') is None:
            continue
        elif medlines.find('MeshHeadingList') is None:
            continue
        elif article_info.find('Language').text != 'eng':
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
                if article_info.find('AuthorList') is None:
                    author_list = 'None'
                else:
                    for author in article_info.find('AuthorList').findall('Author'):
                        if author.find('LastName') is None:
                            name = author.find('CollectiveName').text
                        else:
                            last_name = author.find('LastName').text
                            if author.find('ForeName') is None:
                                name = last_name
                            else:
                                first_name = author.find('ForeName').text
                                name = first_name + ',' + last_name
                        author_list.append(name)
                    if article_info.find('AuthorList').attrib['CompleteYN'] == 'N':
                        author_list.append('et al.')
                for mesh in medlines.find('MeshHeadingList').findall('MeshHeading'):
                    m = mesh.find('DescriptorName').attrib['UI']
                    m_name = mesh.find('DescriptorName').text
                    mesh_list[m] = m_name
                for elocation in article_info.findall('ELocationID'):
                    if elocation.attrib['EIdType'] == 'doi':
                        doi = article_info.find('ELocationID').text
                if medlines.find('ChemicalList') is None:
                    chemicals = 'None'
                else:
                    for chem in medlines.find('ChemicalList').findall('Chemical'):
                        c = chem.find('NameOfSubstance').attrib['UI']
                        c_name = chem.find('NameOfSubstance').text
                        chemicals[c] = c_name
                if medlines.find('SupplMeshList') is None:
                    supply_mesh = 'None'
                else:
                    for supply in medlines.find('SupplMeshList').findall('SupplMeshName'):
                        if supply.attrib['Type'] in supply_mesh:
                            supply_mesh[supply.attrib['Type']].append(supply.text)
                        else:
                            supply_mesh[supply.attrib['Type']] = []
                            supply_mesh[supply.attrib['Type']].append(supply.text)
                data_point['pmid'] = pmid
                data_point['title'] = title
                data_point['abstractText'] = abstract
                data_point["mesh"] = mesh_list
                data_point['authors'] = author_list
                data_point['journal'] = journal_name
                data_point['year'] = year
                data_point['doi'] = doi
                data_point['chemicals'] = chemicals
                data_point['supplMesh'] = supply_mesh
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
