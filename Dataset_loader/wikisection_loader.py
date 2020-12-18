from pathlib import Path
from itertools import chain
import torch
import numpy
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import json
from tqdm import tqdm


stopword = set(stopwords.words("german"))
padding_id = 0
unk_id = 1


class Vocab:
    def __init__(self, init_unk_and_pad=False):
        if init_unk_and_pad:
            self.word_len = 2
            self.word2int = {'unk': unk_id, 'pad': padding_id}
            self.int2word = ['pad', 'unk']

        else:
            self.word_len = 0
            self.word2int = {}
            self.int2word = []
        self.word2frequency = {}

    def add(self, word):
        if word not in self.word2int:
            self.word2int[word] = self.word_len
            self.word_len += 1
            self.int2word.append(word)
        if word not in self.word2frequency:
            self.word2frequency[word] = 1
        else:
            self.word2frequency[word] += 1

    def load(self, path):
        with path.open('r', encoding='UTF-8') as f:
            vocab_dict = json.load(f)
            self.word_len, self.word2int, self.int2word, self.word2frequency = \
                vocab_dict['word_len'], vocab_dict['word2int'], vocab_dict['int2word'], vocab_dict['word2frequency']

    def save(self, path):
        with path.open('w', encoding='UTF-8') as f:
            json.dump({'word_len': self.word_len,
                       'word2int': self.word2int,
                       'int2word': self.int2word,
                        'word2frequency': self.word2frequency}, f, indent=4)


class TestPaperDataset:
    def __init__(self, paper, avg_windows_size, word_vocab, topic_vocab, head_vocab):
        self.paper = paper
        self.avg_windows_size = avg_windows_size
        self.word_vocab = word_vocab
        self.topic_vocab = topic_vocab
        self.head_vocab = head_vocab

    def __len__(self):
        return len(self.paper)

    def __getitem__(self, item):
        return self.paper[item]

    def shuffle(self):
        random.shuffle(self.paper)


def build_vocab(papers_train_path, train_path, rebuild_vocab=True):
    """
    you know, build vocab
    :param train_path: parent path of papers_train_path
    :param rebuild_vocab: if we need to build vocab from initial data or load from file
    :param papers_train_path: list of path contains papers
    :return: word_vocab, topic_vocab, head_vocab
    """
    word_vocab, topic_vocab, head_vocab = Vocab(), Vocab(), Vocab()

    if not rebuild_vocab:
        word_vocab.load(Path(train_path).parent / Path('word_vocab.pt'))
        topic_vocab.load(Path(train_path).parent / Path('topic_vocab.pt'))
        head_vocab.load(Path(train_path).parent / Path('head_vocab.pt'))
        return word_vocab, topic_vocab, head_vocab

    for paper in tqdm(papers_train_path):
        process_paper(paper, head_vocab, topic_vocab, word_vocab)

    word_vocab.save(Path(train_path).parent / Path('word_vocab.pt'))
    topic_vocab.save(Path(train_path).parent / Path('topic_vocab.pt'))
    head_vocab.save(Path(train_path).parent / Path('head_vocab.pt'))
    return word_vocab, topic_vocab, head_vocab


def process_paper(paper, head_vocab, topic_vocab, word_vocab):
    with Path(paper).open('r', encoding='UTF-8') as f:
        lines = f.readlines()
    seperator = '=========='
    for line in lines:
        if seperator in line and len(line) > len(seperator):
            try:
                topic_vocab.add(line.split(';')[1].strip())
                head_vocab.add(line.split(';')[2].strip().split('|')[0].strip())
            except:
                pass

        elif line == '':
            pass

        else:
            [word_vocab.add(word) for word in nltk_seg(line)]


def read_data(path):
    if path is None:
        return None
    files = list(Path(path).glob('**/*.ref'))
    return files


def collate_fn(batch):
    batched_sentence_id = []  # [paper_1, ..., [sentence_1, ..., [word_1, word_2, ...]], ... paper_n]
    batched_attention_mask = []
    batched_label = []  # [paper_1, ..., [sentence_1, ..., sentence_n], ... paper_n]
    batched_topic_label = []
    for paper in batch:
        batched_label.append(torch.tensor(list(chain.from_iterable(paper['label']))))
        batched_topic_label.append(torch.tensor(list(chain.from_iterable(paper['topic_sentence_int']))))
        sentences_id = []
        attention_mask = []
        for i in range(len(paper['paragraph'])):
            for j in range(len(paper['paragraph'][i])):
                sentences_id.append(paper['paragraph'][i][j]['input_ids'])
                attention_mask.append(paper['paragraph'][i][j]['attention_mask'])

        sentences_id_tensor = torch.cat(sentences_id, dim=0)
        attention_mask_tensor = torch.cat(attention_mask, dim=0)
        batched_sentence_id.append(sentences_id_tensor)
        batched_attention_mask.append(attention_mask_tensor)

    return batched_sentence_id, batched_attention_mask, batched_label, batched_topic_label


def nltk_seg(sentence):
    seg_words = word_tokenize(sentence)
    seg_words = [word for word in seg_words if word not in stopword]
    return seg_words


def cal_mean_windows_size(train_path, rebuild_vocab, *papers):
    if not rebuild_vocab:
        with (Path(train_path).parent / Path('paper.pt')).open('r', encoding='UTF-8') as f:
            vocab_dict = json.load(f)
            mean_windows_size = vocab_dict['mean_windows_size']
            return mean_windows_size

    windows_size = []
    seperator = '=========='
    for papers_type in papers:
        for paper in papers_type:
            with Path(paper).open('r', encoding='UTF-8') as f:
                lines = f.readlines()
            segment_len = 0
            for line in lines[1:]:
                if seperator in line:
                    windows_size.append(segment_len)
                    segment_len = 0
                else:
                    segment_len += 1

    with (Path(train_path).parent / Path('paper.pt')).open('w', encoding='UTF-8') as f:
        json.dump({'mean_windows_size': int(numpy.mean(windows_size))}, f, indent=4)

    return int(numpy.mean(windows_size))


def load_wikisection(args):
    train_path, valid_path, test_path = args.train, args.valid, args.test

    papers_train_path, papers_valid_path, papers_test_path = read_data(train_path), read_data(valid_path), read_data(test_path)

    word_vocab, topic_vocab, head_vocab = build_vocab(papers_train_path, train_path, args.rebuild_vocab)

    mean_windows_size = cal_mean_windows_size(train_path, args.rebuild_vocab, papers_train_path, papers_valid_path, papers_test_path)

    train_set, valid_set, test_set = \
        TestPaperDataset(papers_train_path, mean_windows_size, word_vocab, topic_vocab, head_vocab), \
        TestPaperDataset(papers_valid_path, mean_windows_size, word_vocab, topic_vocab, head_vocab), \
        TestPaperDataset(papers_test_path, mean_windows_size, word_vocab, topic_vocab, head_vocab)

    return train_set, valid_set, test_set
