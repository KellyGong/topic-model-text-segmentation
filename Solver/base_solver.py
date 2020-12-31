import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from .solver import Solver
import logging
import numpy as np
from .document import Paper
from pathlib import Path
from tqdm import tqdm
from Evaluation import TopicClassificationMetric, SegmentationMetric
from Model import LSTM_Classifier, Sector_Classifier, TransformerClassifier, S_LSTM_Classifier
from transformers import BertModel, BertTokenizer
from nltk.corpus import stopwords


class BERT_Solver(Solver):
    def __init__(self, args, train_set, valid_set, test_set):
        super(BERT_Solver, self).__init__(args, train_set, valid_set, test_set)
        self.args = args
        self.topic_vocab = train_set.topic_vocab
        self.lda_vocab = train_set.lda_vocab
        self.device = args.device if torch.cuda.is_available() else 'cpu'

        # model
        # self.model = LSTM_Classifier(topic_size=self.topic_vocab.word_len, hidden_dim=self.args.lstm_hidden_size,
        #                              n_layers=self.args.lstm_layer)

        self.model = S_LSTM_Classifier(topic_size=self.topic_vocab.word_len)

        # self.model = TransformerClassifier(topic_size=self.topic_vocab.word_len)

        # move to cuda device
        self.current_epoch = 0

        self.model_need_enforced_teach_and_segment_label = True

        self.model = self.model.to(self.device)

        self.init_optimizer()
        self.init_optimizer_scheduler()

        self.sentence_encoder_dict = self.init_sentence_encoder()

        self.segmentation_loss_function = self.init_segmentation_loss()
        self.topic_classification_loss_function = self.init_topic_classification_loss()

        # init torch seed
        torch.manual_seed(self.seed)

        # balance loss of segmentation and topic classification: segmentation loss + alpha * topic loss
        self.alpha = 0.1

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

    def read_file(self, path, reconstruct=False):
        paper = Paper(path, reconstruct, self.topic_vocab.word2int, self.sentence_encoder_dict,
                      self.device, reproduce_bert=self.args.reproduce_bert, lda_vocab=self.lda_vocab)
        return paper

    def collate_fn(self, batch):
        sentence_embedding_papers = []
        topic_label_papers = []
        segment_label_papers = []
        for path in batch:
            paper = self.read_file(path, reconstruct=self.args.reconstruct if self.current_epoch == 0 else False)
            segment_length = paper.segment_len
            topic_segment_label = paper.topic_label
            sentence_segmentation_label = []
            topic_sentence_label = []
            for i, length_per_segment in enumerate(segment_length):
                sentence_segmentation_label.extend([1] + [0] * (length_per_segment - 1))
                topic_sentence_label.extend([topic_segment_label[i]] * length_per_segment)

            topic_label_papers.append(topic_sentence_label)
            segment_label_papers.append(sentence_segmentation_label)
            sentence_embedding_papers.append(torch.from_numpy(paper.bert_embedding))

        return sentence_embedding_papers, topic_label_papers, segment_label_papers

    def load_dataset(self, dataset, shuffle=False, drop_last=False):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle,
                                 collate_fn=self.collate_fn, drop_last=drop_last, num_workers=8, prefetch_factor=4)
        return data_loader

    def train_and_evaluate(self):
        self.train()
        self.evaluate()

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

        begin_epoch = result_parameter['epoch']

        for epoch in range(begin_epoch, self.epoch):
            self.current_epoch = epoch
            train_loader = self.load_dataset(self.train_set, shuffle=True, drop_last=True)
            valid_loader = self.load_dataset(self.valid_set, shuffle=False)

            loss = self.train_or_evaluate_epoch(train_loader, epoch, mode='train')
            topic_classification_metric, segmentation_metric = \
                self.train_or_evaluate_epoch(valid_loader, epoch, mode='validation')

            if topic_classification_metric.get_map() > result_parameter['MAP']:
                result_parameter['MAP'] = topic_classification_metric.get_map()
                result_parameter['f1'] = topic_classification_metric.get_precision1()
                result_parameter['epoch'] = epoch
                result_parameter['Pk'] = segmentation_metric.get_pk()
                result_parameter['loss'] = loss
                self.save_model(epoch, loss, result_parameter['Pk'], result_parameter['f1'],
                                result_parameter['MAP'], 'best_model.pt')

            self.save_model(epoch, loss, segmentation_metric.get_pk(), topic_classification_metric.get_map(),
                            topic_classification_metric.get_precision1(), 'last_model.pt')

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

    def train_or_evaluate_epoch(self, data_loader, epoch, mode='train'):

        topic_classification_metric = TopicClassificationMetric()
        segmentation_metric = SegmentationMetric(k=self.train_set.avg_windows_size)

        if mode == 'train':
            self.model.train()
            epoch_loss = .0
            epoch_loss_segmentation = .0
            epoch_loss_topic_classification = .0

            for sentence_embedding_papers, topic_label_papers, segment_label_papers in tqdm(data_loader):
                self.optimizer.zero_grad()
                step_loss_segmentation = torch.FloatTensor([.0]).to(self.device)
                step_loss_topic_classification = torch.FloatTensor([.0]).to(self.device)

                sentence_embedding_papers_cuda = [sentence_embedding_paper.to(self.device)
                                                  for sentence_embedding_paper in sentence_embedding_papers]

                if not self.model_need_enforced_teach_and_segment_label:
                    segmentation_probability, segmentation_identification, topic_classification_probability = self.model(
                        sentence_embedding_papers_cuda)

                else:
                    tensor_segment_label_papers = [torch.tensor(segment_label_paper, dtype=torch.long,
                                                   device=self.device) for segment_label_paper in segment_label_papers]

                    segmentation_probability, segmentation_identification, topic_classification_probability = self.model(
                        sentence_embedding_papers_cuda, tensor_segment_label_papers, True if epoch < self.enforced_teach else False)

                for (segmentation_probability_paper, segmentation_identification_paper,
                     topic_classification_probability_paper, topic_label_paper, segment_label_paper) in \
                        zip(segmentation_probability, segmentation_identification, topic_classification_probability,
                            topic_label_papers, segment_label_papers):
                    step_loss_segmentation += \
                        self.segmentation_loss_function(segmentation_probability_paper,
                                                        torch.from_numpy(np.array(segment_label_paper)).float().to(
                                                            self.device))
                    step_loss_topic_classification += \
                        self.topic_classification_loss_function(topic_classification_probability_paper,
                                                                torch.from_numpy(np.array(topic_label_paper)).long().to(
                                                                    self.device))

                    self.update_segmentation_metric(segmentation_metric, segmentation_identification_paper,
                                                    segment_label_paper)
                    self.update_topic_classification_metric(topic_classification_metric,
                                                            F.softmax(topic_classification_probability_paper, dim=-1), topic_label_paper,
                                                            segmentation_identification_paper, segment_label_paper)

                step_loss = step_loss_segmentation + self.alpha * step_loss_topic_classification
                epoch_loss += step_loss.detach().item()
                epoch_loss_segmentation += step_loss_segmentation.detach().item()
                epoch_loss_topic_classification += step_loss_topic_classification.detach().item()

                step_loss.backward()
                self.optimizer.step()

            logging.info(f'{mode} epoch: {epoch}, loss: {round(epoch_loss, 4)}, '
                         f'segmentation_loss: {round(epoch_loss_segmentation, 4)}, '
                         f'topic_classification_loss: {round(epoch_loss_topic_classification, 4)}, '
                         f'Pk: {round(segmentation_metric.get_pk(), 4)}, '
                         f'F1: {round(topic_classification_metric.get_precision1(), 4)}, '
                         f'MAP: {round(topic_classification_metric.get_map(), 4)}')

            return epoch_loss

        elif mode == 'test' or 'validation':
            self.model.eval()
            for sentence_embedding_papers, topic_label_papers, segment_label_papers in tqdm(data_loader):

                sentence_embedding_papers_cuda = [sentence_embedding_paper.to(self.device)
                                                  for sentence_embedding_paper in sentence_embedding_papers]

                if not self.model_need_enforced_teach_and_segment_label:
                    segmentation_probability, segmentation_identification, topic_classification_probability = self.model(
                        sentence_embedding_papers_cuda)

                else:
                    tensor_segment_label_papers = [torch.tensor(segment_label_paper, dtype=torch.long,
                                                                device=self.device) for segment_label_paper in
                                                   segment_label_papers]

                    segmentation_probability, segmentation_identification, topic_classification_probability = self.model(
                        sentence_embedding_papers_cuda, tensor_segment_label_papers, False)

                for (segmentation_probability_paper, segmentation_identification_paper,
                     topic_classification_probability_paper, topic_label_paper, segment_label_paper) in \
                        zip(segmentation_probability, segmentation_identification, topic_classification_probability,
                            topic_label_papers, segment_label_papers):
                    self.update_segmentation_metric(segmentation_metric, segmentation_identification_paper,
                                                    segment_label_paper)
                    self.update_topic_classification_metric(topic_classification_metric,
                                                            F.softmax(topic_classification_probability_paper, dim=-1), topic_label_paper,
                                                            segmentation_identification_paper, segment_label_paper)

            logging.info(f'{mode} epoch: {epoch}, '
                         f'Pk: {round(segmentation_metric.get_pk(), 4)}, '
                         f'F1: {round(topic_classification_metric.get_precision1(), 4)}, '
                         f'MAP: {round(topic_classification_metric.get_map(), 4)}')

            return topic_classification_metric, segmentation_metric

        else:
            raise NotImplementedError

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

            logging.info('load last model success !')
            logging.info(f'continue training from {model["epoch"]} epoch !')

            return train_parameters

        else:
            return None
