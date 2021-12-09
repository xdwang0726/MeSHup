import argparse
import json


def sort_list_dic(file):
    with open(file, 'rb') as infile:
        articles = json.load(infile)['articles']
        sorted_articles = sorted(articles, key=lambda d: d['pmid'])
    return sorted_articles


def remove_duplicate(dict):
    result = {}

    for key, value in dict.items():
        if key not in result.values():
            result[key] = value
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pubmed')
    parser.add_argument('--pmc')
    parser.add_argument('--save')

    args = parser.parse_args()

    sorted_pubmed = sort_list_dic(args.pubmed)
    sorted_pmc = sort_list_dic(args.pmc)

    print('check if pubmed articles and pmc articles are corresponded')
    assert len(sorted_pubmed) == len(sorted_pmc)

    results = []
    for i, pubmed in sorted_pubmed:
        if pubmed['pmid'] == sorted_pmc[i]['pmid']:
            article = {**pubmed, **sorted_pmc[i]}
            article = remove_duplicate(article)
            results.append(article)

    pubmed = {'articles': results}
    with open(args.save, "w") as outfile:
        json.dump(pubmed, outfile, indent=4)


if __name__ == "__main__":
    main()


