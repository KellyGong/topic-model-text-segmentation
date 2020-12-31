import torch
import torch.nn.functional as F
from .solver import Solver
import logging
import numpy as np
from tqdm import tqdm
from Evaluation import TopicClassificationMetric, SegmentationMetric
from Model import LDA_LSTM_Classifier, LDA_LSTM_Classifier_2
from .early_stop import EarlyStop


class NVI_BERT_Solver(Solver):
    def __init__(self, args, train_set, valid_set, test_set):
        super(NVI_BERT_Solver, self).__init__(args, train_set, valid_set, test_set)

        # model
        self.model = LDA_LSTM_Classifier_2(args=args, topic_size=self.topic_vocab.word_len,
                                           lda_word_size=self.lda_vocab.word_len,
                                           hidden_dim=self.args.lstm_hidden_size,
                                           n_layers=self.args.lstm_layer).to(self.device)
        self.init_optimizer()
        self.init_optimizer_scheduler()

    def collate_fn(self, batch):
        sentence_embedding_papers = []
        topic_label_papers = []
        segment_label_papers = []
        lda_sentence_papers = []
        for path in batch:

            paper = self.read_file(path, reconstruct=self.args.reconstruct if self.current_epoch == 0 else False)
            segment_length = paper.segment_len
            if sum(segment_length) < 2:
                continue
            topic_segment_label = paper.topic_label
            sentence_segmentation_label = []
            topic_sentence_label = []
            for i, length_per_segment in enumerate(segment_length):
                sentence_segmentation_label.extend([1] + [0] * (length_per_segment - 1))
                topic_sentence_label.extend([topic_segment_label[i]] * length_per_segment)

            topic_label_papers.append(topic_sentence_label)
            segment_label_papers.append(sentence_segmentation_label)
            sentence_embedding_papers.append(torch.from_numpy(paper.bert_embedding))
            lda_sentence_papers.append(torch.from_numpy(self.lda_int2lda_one_hot(paper.lda_sentence)).float())

        return sentence_embedding_papers, topic_label_papers, segment_label_papers, lda_sentence_papers

    def lda_int2lda_one_hot(self, lda_int):
        sentences_one_hot = []
        for segment in lda_int:
            for sentence in segment:
                sentences_one_hot.append(np.bincount(np.array(sentence).astype(int), minlength=self.lda_vocab.word_len))

        return np.array(sentences_one_hot).astype(int)

    def train_and_evaluate(self):
        self.train()
        self.evaluate()

    def train_or_evaluate_epoch(self, data_loader, epoch, mode='train'):

        topic_classification_metric = TopicClassificationMetric()
        segmentation_metric = SegmentationMetric(k=self.train_set.avg_windows_size)

        if mode == 'train':
            self.model.train()
            epoch_loss = .0
            epoch_loss_segmentation = .0
            epoch_loss_topic_classification = .0
            epoch_loss_lda = .0
            for sentence_embedding_papers, topic_label_papers, segment_label_papers, lda_one_hot_papers in tqdm(
                    data_loader):
                sentence_embedding_papers_cuda = \
                    [sentence_embedding_paper.to(self.device) for sentence_embedding_paper in sentence_embedding_papers]
                lda_one_hot_papers_cuda = [lda_one_hot_paper.to(self.device) for lda_one_hot_paper in
                                           lda_one_hot_papers]

                self.optimizer.zero_grad()
                step_loss_segmentation = torch.FloatTensor([.0]).to(self.device)
                step_loss_topic_classification = torch.FloatTensor([.0]).to(self.device)
                step_loss_lda = torch.FloatTensor([.0]).to(self.device)
                if not self.model_need_enforced_teach_and_segment_label:
                    segmentation_probability, segmentation_identification, topic_classification_probability = self.model(
                        sentence_embedding_papers)

                else:
                    tensor_segment_label_papers = [torch.tensor(segment_label_paper, dtype=torch.long,
                                                                device=self.device) for segment_label_paper in
                                                   segment_label_papers]

                    segmentation_probability, segmentation_identification, topic_classification_probability, lda_loss = self.model(
                        sentence_embedding_papers_cuda, lda_one_hot_papers_cuda, tensor_segment_label_papers,
                        True if epoch < self.enforced_teach else False)

                for (segmentation_probability_paper, segmentation_identification_paper,
                     topic_classification_probability_paper, topic_label_paper, segment_label_paper, lda_loss_paper) in \
                        zip(segmentation_probability, segmentation_identification, topic_classification_probability,
                            topic_label_papers, segment_label_papers, lda_loss):
                    step_loss_segmentation += \
                        self.segmentation_loss_function(segmentation_probability_paper,
                                                        torch.from_numpy(np.array(segment_label_paper)).float().to(
                                                            self.device))
                    step_loss_topic_classification += \
                        self.topic_classification_loss_function(topic_classification_probability_paper,
                                                                torch.from_numpy(np.array(topic_label_paper)).long().to(
                                                                    self.device))

                    step_loss_lda += lda_loss_paper

                    self.update_segmentation_metric(segmentation_metric, segmentation_identification_paper,
                                                    segment_label_paper)
                    self.update_topic_classification_metric(topic_classification_metric,
                                                            F.softmax(topic_classification_probability_paper, dim=-1),
                                                            topic_label_paper,
                                                            segmentation_identification_paper, segment_label_paper)

                step_loss = step_loss_segmentation + self.alpha * step_loss_topic_classification + self.beta * step_loss_lda
                epoch_loss += step_loss.detach().item()
                epoch_loss_segmentation += step_loss_segmentation.detach().item()
                epoch_loss_topic_classification += step_loss_topic_classification.detach().item()
                epoch_loss_lda += step_loss_lda.detach().item()

                step_loss.backward()
                self.optimizer.step()

            logging.info(f'{mode} epoch: {epoch}, loss: {round(epoch_loss, 4)}, '
                         f'segmentation_loss: {round(epoch_loss_segmentation, 4)}, '
                         f'topic_classification_loss: {round(epoch_loss_topic_classification, 4)}, '
                         f'lda_loss: {round(epoch_loss_lda, 4)}, '
                         f'Pk: {round(segmentation_metric.get_pk(), 4)}, '
                         f'F1: {round(topic_classification_metric.get_precision1(), 4)}, '
                         f'MAP: {round(topic_classification_metric.get_map(), 4)}')

            return epoch_loss

        elif mode == 'test' or 'validation':
            self.model.eval()
            for sentence_embedding_papers, topic_label_papers, segment_label_papers, lda_one_hot_papers in tqdm(
                    data_loader):
                sentence_embedding_papers_cuda = \
                    [sentence_embedding_paper.to(self.device) for sentence_embedding_paper in sentence_embedding_papers]
                lda_one_hot_papers_cuda = [lda_one_hot_paper.to(self.device) for lda_one_hot_paper in
                                           lda_one_hot_papers]

                tensor_segment_label_papers = [torch.tensor(segment_label_paper, dtype=torch.long,
                                                            device=self.device) for segment_label_paper in
                                               segment_label_papers]

                segmentation_probability, segmentation_identification, topic_classification_probability, _ = self.model(
                    sentence_embedding_papers_cuda, lda_one_hot_papers_cuda, tensor_segment_label_papers, False)

                for (segmentation_probability_paper, segmentation_identification_paper,
                     topic_classification_probability_paper, topic_label_paper, segment_label_paper) in \
                        zip(segmentation_probability, segmentation_identification, topic_classification_probability,
                            topic_label_papers, segment_label_papers):
                    self.update_segmentation_metric(segmentation_metric, segmentation_identification_paper,
                                                    segment_label_paper)
                    self.update_topic_classification_metric(topic_classification_metric,
                                                            F.softmax(topic_classification_probability_paper, dim=-1),
                                                            topic_label_paper,
                                                            segmentation_identification_paper, segment_label_paper)

            logging.info(f'{mode} epoch: {epoch}, '
                         f'Pk: {round(segmentation_metric.get_pk(), 4)}, '
                         f'F1: {round(topic_classification_metric.get_precision1(), 4)}, '
                         f'MAP: {round(topic_classification_metric.get_map(), 4)}')

            return topic_classification_metric, segmentation_metric

        else:
            raise NotImplementedError


