# MeSHup: Corpus for Full Text Biomedical Document Indexing
Medical Subject Heading (MeSH) indexing refers to the problem of assigning each given biomedical document with the most relevant labels from an extremely large set of MeSH terms. Currently, the vast number of biomedical articles in the PubMed database are manually annotated by human curators, which is time consuming and costly; therefore, a computational system that can assist the indexing is highly valuable. When developing supervised MeSH indexing systems, the availability of a large-scale annotated text corpus is desirable. A publicly available, large corpus that permits robust  evaluation and comparison of various systems is important to the research community. We release a large scale annotated MeSH indexing corpus, MeSHup, which contains 1,342,667 full text articles, associated MeSH labels and metadata, such authors and publish venues, that are collected from the MEDLINE database. We train a end-to-end model that combines features from  documents  and their associated labels on our corpus and report the new baseline.
## Download Dataset

## Required Packages
- Python 3.7
- numpy==1.11.1
- dgl-gpu==0.6.1
- nltk==3.5
- scikit-learn==0.23.0
- scipy==1.4.1
- sklearn==0.0
- spacy==2.2.2
- tokenizers==0.9.3
- torch==1.6.0
- torchtext==0.6.0
- tqdm==4.60.0
- transformers==3.5.1

We use the BioWordVec as our embeddings 

## Usage
### Get the label graph
python -u build_graph.py --word2vec_path PATH_TO_EMBEDDINGS --meSH_pair_path data/mesh/MeSH_id_pair.txt --mesh_parent_children_path data/mesh/MeSH_parent_children_mapping.txt --output ../graph.bin
### Training 
python -u run_classifier.py --full_path full.csv --train_path train.csv --test_path test.csv --dev_path /home/xdwang/scratch/PMC/pmc/dev.csv --word2vec_path BioWord2Vec_standard.w2v meSH_pair_path data/mesh/MeSH_id_pair.txt --graph ../graph.bin --save-model ../model.bin


