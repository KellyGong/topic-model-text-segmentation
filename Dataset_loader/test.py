import json

from pathlib import Path


path = Path('../data/wikisection_dataset_ref/en_city/head_vocab.pt')
with path.open('r', encoding='UTF-8') as f:
    vocab_dict = json.load(f)['word2frequency']

    filter_dict = {k: v for k, v in vocab_dict.items() if v > 2}
    print(len(filter_dict))
    print(len(vocab_dict))