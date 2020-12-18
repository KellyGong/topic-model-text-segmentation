import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


class LSTM_Classifier(nn.Module):
    def __init__(self, topic_size, GRU=False, input_dim=768, hidden_dim=128, output_dim=2, n_layers=2,
                 bidirectional=True,
                 dropout=0.5):
        super(LSTM_Classifier, self).__init__()
        if GRU:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                              dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                               dropout=dropout, batch_first=True)

        self.topic_size = topic_size
        self.Ws_segmentation = nn.Parameter(torch.Tensor(hidden_dim * 2, output_dim))
        self.bs_segmentation = nn.Parameter(torch.zeros((output_dim,)))

        self.softmax = F.softmax
        self.top = 0.7

        self.Ws_topic_classification = nn.Parameter(torch.Tensor(hidden_dim * 2, topic_size))
        self.bs_topic_classification = nn.Parameter(torch.zeros((topic_size,)))

        nn.init.uniform_(self.Ws_segmentation, -0.1, 0.1)
        nn.init.uniform_(self.Ws_topic_classification, -0.1, 0.1)

    def pad(self, sentences_embedding_papers):
        len_paper_list = [paper.size(0) for paper in sentences_embedding_papers]
        return pad_sequence(sentences_embedding_papers, batch_first=True, padding_value=0), len_paper_list

    def forward(self, sentences_embedding_papers):

        # model transfer
        padded_sentences_embedding_papers, len_paper_list = self.pad(sentences_embedding_papers)
        padded_sentences_embedding_after_rnn, (_) = self.rnn(padded_sentences_embedding_papers)
        padded_segment_probability = self.softmax(padded_sentences_embedding_after_rnn.matmul(self.Ws_segmentation)
                                                  + self.bs_segmentation, dim=-1)
        padded_segment_label = \
            torch.ge(padded_segment_probability,
                     torch.full_like(padded_segment_probability, self.top)).long().chunk(2, -1)[0].squeeze(-1)

        padded_topic_probability = \
            padded_sentences_embedding_after_rnn.matmul(self.Ws_topic_classification) + self.bs_topic_classification

        # deal with result same shape with document
        segment_label_papers = [segment_label_per_paper.squeeze(0)[:len_paper] for (segment_label_per_paper, len_paper)
                                in
                                zip(padded_segment_label.chunk(padded_segment_label.size(0), 0), len_paper_list)]
        padded_segment_probability = padded_segment_probability.squeeze(1).chunk(2, -1)[0].squeeze(-1)
        segment_probability_papers = [res_pro_tensor.squeeze(0)[:len_paper] for (res_pro_tensor, len_paper) in
                                      zip(padded_segment_probability.chunk(padded_segment_probability.size(0), 0), len_paper_list)]
        topic_probability_papers = [topic_probability_paper.squeeze(0)[:len_paper] for
                                    (topic_probability_paper, len_paper) in
                                    zip(padded_topic_probability.chunk(padded_topic_probability.size(0), 0), len_paper_list)]

        return segment_probability_papers, segment_label_papers, topic_probability_papers
