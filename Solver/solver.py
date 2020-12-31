import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
import torch
from transformers import BertModel, BertTokenizer
from nltk.corpus import stopwords
from .document import Paper
from torch.utils.data import DataLoader
from Evaluation import TopicClassificationMetric, SegmentationMetric
from pathlib import Path
import numpy as np
import logging
from .early_stop import EarlyStop


class Solver:
    def __init__(self, args, train_set=None, valid_set=None, test_set=None):
        self.args = args
        self.lr = args.lr
        self.optimizer = args.optimizer
        self.batch_size = args.batch_size
        self.lr_scheduler = args.lr_scheduler
        self.enforced_teach = args.enforced_teach
        self.factor = args.factor
        self.min_lr = args.min_lr
        self.save_path = args.save_path
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.seed = args.seed
        self.epoch = args.epoch
        self.model = None

        self.topic_vocab = train_set.topic_vocab
        self.lda_vocab = train_set.lda_vocab
        self.device = args.device if torch.cuda.is_available() else 'cpu'

        self.model_need_enforced_teach_and_segment_label = True

        self.sentence_encoder_dict = self.init_sentence_encoder()

        self.segmentation_loss_function = self.init_segmentation_loss()
        self.topic_classification_loss_function = self.init_topic_classification_loss()

        # init torch seed
        torch.manual_seed(self.seed)

        # begin_epoch
        self.current_epoch = 0

        # balance loss of segmentation and topic classification: segmentation loss + alpha * topic loss
        self.alpha = 0.1
        self.beta = 0.1

    def init_optimizer(self):
        if self.optimizer == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        else:
            return NotImplementedError

    def init_optimizer_scheduler(self):
        if self.lr_scheduler == 'None':
            return None

        elif self.lr_scheduler == 'Plateau':
            assert 0 < self.factor < 1
            assert self.min_lr < self.lr
            self.lr_scheduler = ReduceLROnPlateau(self.optimizer, 'min', factor=self.factor, min_lr=self.min_lr, patience=3)

        else:
            raise NotImplementedError

    def init_segmentation_loss(self):
        return nn.BCELoss(reduction='sum').to(self.device)

    def init_topic_classification_loss(self):
        return nn.CrossEntropyLoss(reduction='sum').to(self.device)

    def init_sentence_encoder(self):
        if self.args.language == 'EN':
            sentence_encoder = BertModel.from_pretrained('bert-base-uncased').to(self.device)
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
            stopwords_dic = set(stopwords.words("english"))

        elif self.args.language == 'DE':
            sentence_encoder = BertModel.from_pretrained('bert-base-german-cased').to(self.device)
            tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
            stopwords_dic = set(stopwords.words("german"))

        else:
            raise NotImplementedError

        for param in sentence_encoder.parameters():
            param.requires_grad = False

        return {
            'sentence_encoder': sentence_encoder,
            'tokenizer': tokenizer,
            'stopwords': stopwords_dic
        }

    def reload_best_model_and_update_lr(self, lr):
        load_parameter = self.load_model('best_model.pt')

        if not load_parameter:
            raise FileNotFoundError

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        if lr <= self.min_lr:
            logging.info(f'reduce lr to {round(lr, 8)}, stop training!')
            return False

        else:
            logging.info(f'reduce lr to {round(lr, 8)}, continue training!')
            return True

    def read_file(self, path, reconstruct=False):
        paper = Paper(path, reconstruct, self.topic_vocab.word2int, self.sentence_encoder_dict,
                      self.device, reproduce_bert=self.args.reproduce_bert, lda_vocab=self.lda_vocab)
        return paper

    def load_dataset(self, dataset, shuffle=False, drop_last=False):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                                 collate_fn=self.collate_fn, drop_last=drop_last,
                                 pin_memory=False, num_workers=0)
        return data_loader

    @staticmethod
    def update_topic_classification_metric(metric: TopicClassificationMetric, topic_classification_probability_paper,
                                           topic_label_paper, segmentation_identification_paper, segment_label_paper):
        """
        update topic_classification_metric
        :param topic_classification_probability_paper: topic_probability of sentence level
        :param topic_label_paper: [4, 4, 4, 6, 6, 6]
        :param segmentation_identification_paper: same as segment label paper
        :param segment_label_paper: [1, 0, 0, 1, 0, 0]
        """

        def find_max_overlap_segment(segment_reference, segment_hypothesis, order_of_segment):
            """
            :param segment_reference: [1, 0, 0, 1, 0, 0]
            :param order_of_segment: the ith segment: 1 <= i <= number of segments
            :return: beginning and end position of the max overlap segment  (begin, end)
            """

            segment_reference_position = np.cumsum(segment_reference)

            segment_reference_position_add_end = \
                np.append(segment_reference_position, segment_reference_position[-1] + 1)

            begin_position_of_reference = -1
            end_position_of_reference = -1

            for sentence_iterator in range(len(segment_reference_position_add_end)):
                if segment_reference_position_add_end[sentence_iterator] == order_of_segment and \
                        (sentence_iterator == 0 or
                         segment_reference_position_add_end[sentence_iterator - 1] != order_of_segment):
                    begin_position_of_reference = sentence_iterator

                if segment_reference_position_add_end[sentence_iterator] == order_of_segment + 1 and \
                        (segment_reference_position_add_end[sentence_iterator - 1] == order_of_segment):
                    end_position_of_reference = sentence_iterator

            if begin_position_of_reference == -1 or end_position_of_reference == -1 or begin_position_of_reference > end_position_of_reference:
                raise ValueError

            begin_position_max_overlap_segment = -1
            end_position_max_overlap_segment = 0
            max_overlap_length = 0

            masses_array = SegmentationMetric.get_masses_array(segment_hypothesis)

            begin_position = 0
            for masses_array_iterator in masses_array:
                end_position = begin_position + masses_array_iterator
                overlap_begin = max(begin_position, begin_position_of_reference)
                overlap_end = min(end_position, end_position_of_reference)
                overlap_length = max(0, overlap_end - overlap_begin)
                if overlap_length > max_overlap_length:
                    begin_position_max_overlap_segment = begin_position
                    end_position_max_overlap_segment = end_position
                    max_overlap_length = overlap_length
                begin_position += masses_array_iterator

            return begin_position_max_overlap_segment, end_position_max_overlap_segment

        number_of_segment = np.sum(segment_label_paper)
        lengths_of_segment = SegmentationMetric.get_masses_array(segment_label_paper)

        # change first position to 1, document begin must be beginning of a segment
        segmentation_identification_paper_adjust = segmentation_identification_paper.cpu().numpy()
        segmentation_identification_paper_adjust[0] = 1

        segment_topic_label_position = 0

        # segment_label_paper = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0]
        # segmentation_identification_paper_adjust = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0])

        for i in range(1, number_of_segment + 1):
            begin, end = find_max_overlap_segment(np.array(segment_label_paper),
                                                  segmentation_identification_paper_adjust, i)
            segment_topic_distribution = \
                np.average(topic_classification_probability_paper.detach().cpu().numpy()[begin: end][:], axis=0)

            segment_topic_label = topic_label_paper[segment_topic_label_position]
            segment_topic_label_position += lengths_of_segment[i - 1]
            segment_topic_label_binary = np.zeros_like(segment_topic_distribution)
            segment_topic_label_binary[segment_topic_label] = 1

            metric.update(segment_topic_label_binary, segment_topic_distribution)

    @staticmethod
    def update_segmentation_metric(metric: SegmentationMetric, segmentation_identification_paper, segment_label_paper):
        metric.update(segment_label_paper, segmentation_identification_paper.cpu())

    def save_model(self, epoch, loss, pk, f1, MAP, MODEL: str):
        model = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'epoch': epoch,
            'loss': loss,
            'Pk': pk,
            'f1': f1,
            'MAP': MAP
        }

        torch.save(model, Path(self.save_path) / MODEL)

    def load_model(self, MODEL: str):
        model_path = Path(Path(self.save_path) / MODEL)
        if model_path.exists():
            model = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(model['model'])
            self.optimizer.load_state_dict(model['optimizer'])
            self.lr_scheduler.load_state_dict(model['lr_scheduler'])
            train_parameters = {
                'loss': model['loss'],
                'epoch': model['epoch'],
                'Pk': model['Pk'],
                'f1': model['f1'],
                'MAP': model['MAP']
            }

            logging.info('load model success !')
            logging.info(f'continue training from {model["epoch"]} epoch !')

            return train_parameters

        else:
            return None

    def train(self):
        assert self.train_set and self.valid_set is not None
        result_parameter = {
            'epoch': 0,
            'loss': 1e10,
            'Pk': 1e10,
            'f1': 0,
            'MAP': 0
        }

        load_parameter = self.load_model('last_model.pt')
        if load_parameter:
            result_parameter = load_parameter

        early_stop = EarlyStop(lr=self.optimizer.param_groups[0]['lr'], val_metric_max=result_parameter['MAP'], factor=self.factor)

        begin_epoch = result_parameter['epoch']

        for epoch in range(begin_epoch, self.epoch):
            self.current_epoch = epoch

            train_loader = self.load_dataset(self.train_set, shuffle=True, drop_last=True)
            valid_loader = self.load_dataset(self.valid_set, shuffle=False)
            test_loader = self.load_dataset(self.test_set, shuffle=False)

            loss = self.train_or_evaluate_epoch(train_loader, epoch, mode='train')
            topic_classification_metric_valid, segmentation_metric_valid = \
                self.train_or_evaluate_epoch(valid_loader, epoch, mode='validation')
            topic_classification_metric_test, segmentation_metric_test = \
                self.train_or_evaluate_epoch(test_loader, epoch, mode='test')

            if topic_classification_metric_test.get_map() > result_parameter['MAP']:
                result_parameter['MAP'] = topic_classification_metric_test.get_map()
                result_parameter['f1'] = topic_classification_metric_test.get_precision1()
                result_parameter['epoch'] = epoch
                result_parameter['Pk'] = segmentation_metric_test.get_pk()
                result_parameter['loss'] = loss
                self.save_model(epoch, loss, result_parameter['Pk'], result_parameter['f1'],
                                result_parameter['MAP'], 'best_model.pt')

            self.save_model(epoch, loss, segmentation_metric_valid.get_pk(),
                            topic_classification_metric_valid.get_map(),
                            topic_classification_metric_valid.get_precision1(), 'last_model.pt')

            if not early_stop.update(topic_classification_metric_test.get_map()):
                if not self.reload_best_model_and_update_lr(early_stop.lr):
                    break

    def train_or_evaluate_epoch(self, data_loader, epoch, mode):
        raise NotImplementedError

    def evaluate(self):
        assert self.save_path and self.test_set is not None

        load_parameter = self.load_model('best_model.pt')
        if not load_parameter:
            raise FileNotFoundError
        else:
            result_parameter = load_parameter

        test_loader = self.load_dataset(self.test_set, shuffle=False)
        topic_classification_metric, segmentation_metric = \
            self.train_or_evaluate_epoch(test_loader, load_parameter['epoch'], mode='test')

        result_parameter['Pk'] = segmentation_metric.get_pk()
        result_parameter['f1'] = topic_classification_metric.get_precision1()
        result_parameter['MAP'] = topic_classification_metric.get_map()

        return result_parameter



