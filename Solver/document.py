"""
contain data for training
"""
from pathlib import Path
import numpy as np
import json
from nltk.tokenize import word_tokenize
import torch


class Paper:

    def __init__(self, path, reconstruct, topic2int, sentence_encoder, device):
        self.path = path
        self.paragraph = []
        self.topic_label = []
        self.bert_embedding = []
        self.segment_len = []
        self.tokenize_sentence = []
        self.topic2int = topic2int
        self.reconstruct = reconstruct

        self.stopwords_dict = sentence_encoder['stopwords']
        self.sentence_encoder = sentence_encoder['sentence_encoder']
        self.tokenizer = sentence_encoder['tokenizer']
        self.device = device

        self.read_data_from_wikisection()

        assert len(self.topic_label) == len(self.segment_len)

    def nltk_seg(self, sentence):
        seg_words = word_tokenize(sentence)
        seg_words = [word for word in seg_words if word not in self.stopwords_dict]
        return seg_words

    def read_data_from_wikisection(self):
        bert_dict = self.load_bert_presentation_from_file()
        if bert_dict['exist'] is False or self.reconstruct:
            sentence_embedding, label = self.read_raw_data_from_wikisection()
        else:
            sentence_embedding, label = bert_dict['sentence_embedding'], bert_dict['label']

        self.bert_embedding = sentence_embedding
        self.topic_label = label['topic_label']
        self.segment_len = label['segment_len']
        self.tokenize_sentence = label['split_sentence']

        return sentence_embedding, label

    def read_raw_data_from_wikisection(self):
        seperator = '=========='

        with Path(self.path).open('r', encoding='UTF-8') as f:
            lines = f.readlines()

        paragraphs = []
        paragraph = []
        topic_label = []

        for line in lines:
            if seperator in line:
                if len(paragraph) != 0:
                    paragraphs.append(paragraph.copy())
                    paragraph = []
                else:
                    if len(topic_label) > 0:
                        topic_label.pop()
                try:
                    topic_label.append(self.topic2int[line.split(';')[1].strip()])
                except:
                    pass

            elif line == '':
                pass
            elif line != '\n' and len(line) > 0:
                paragraph.append(line.strip())

        assert len(topic_label) == len(paragraphs)

        label = {'segment_len': [], 'topic_label': topic_label, 'split_sentence': []}

        paper_bert_tokenized_sentences = []
        for paragraph in paragraphs:
            split_sentences = [self.nltk_seg(s) for s in paragraph if len(self.nltk_seg(s)) > 0]
            if len(split_sentences) == 0:
                continue
            label['split_sentence'].append(split_sentences)

            split_sentences_bert = [
                self.tokenizer.encode_plus(s, add_special_tokens=True, max_length=64, padding='max_length',
                                           truncation=True, return_attention_mask=True, return_tensors='pt')
                for s in paragraph]

            paper_bert_tokenized_sentences.extend(split_sentences_bert)

            label['segment_len'].append(len(split_sentences_bert))

        # concat sentence id to tensor, and concat attention mask to tensor
        sentence_id, attention_mask = [], []
        for bert_tokenized_sentence in paper_bert_tokenized_sentences:
            sentence_id.append(bert_tokenized_sentence['input_ids'])
            attention_mask.append(bert_tokenized_sentence['attention_mask'])

        sentence_id_tensor = torch.cat(sentence_id, dim=0)
        attention_mask_tensor = torch.cat(attention_mask, dim=0)

        bert_embedding = self.sentence_encoder(sentence_id_tensor.to(self.device), attention_mask=attention_mask_tensor.to(self.device),
                                       token_type_ids=None, return_dict=True).pooler_output.detach().cpu().numpy()

        assert len(self.topic_label) == len(self.paragraph)

        self.save_bert_presentation_to_file(bert_embedding, label)

        return bert_embedding, label

    def save_bert_presentation_to_file(self, bert_embedding, label_dict):
        data_path = Path(self.path)
        bert_presentation_path = data_path.parent / (data_path.name + '.npy')
        label_path = data_path.parent / (data_path.name + '.json')
        with label_path.open('w', encoding='UTF-8') as f:
            json.dump(label_dict, f)
        np.save(bert_presentation_path, bert_embedding)

    def load_bert_presentation_from_file(self):
        """
        load bert output from ****.npy
        load label information from ****.json
        :param path:  str ****.ref
        :return: sentence embedding saved in ****.npy, paper label saved in ****.npy
        """
        data_path = Path(self.path)
        bert_presentation_path = data_path.parent / (data_path.name + '.npy')
        label_path = data_path.parent / (data_path.name + '.json')

        if bert_presentation_path.exists() and label_path.exists():
            sentence_embedding = np.load(bert_presentation_path)
            with label_path.open('r', encoding='UTF-8') as f:
                label = json.loads(f.read())

            return {
                'exist': True,
                'sentence_embedding': sentence_embedding,
                'label': label
            }

        else:
            return {
                'exist': False
            }


