"""

combined neural variational inference network with

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from NVI_Model import ProdLDA


class LDA_LSTM_Classifier(nn.Module):
    def __init__(self, args, topic_size, lda_word_size, GRU=False, input_dim=768, hidden_dim=128, output_dim=2, n_layers=2,
                 bidirectional=True,
                 dropout=0.5):
        super(LDA_LSTM_Classifier, self).__init__()
        if GRU:
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                              dropout=dropout, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                               dropout=dropout, batch_first=True)

        self.lda = ProdLDA(args, lda_word_size)

        self.topic_size = topic_size
        self.Ws_segmentation = nn.Parameter(torch.Tensor(hidden_dim * 2 + args.num_topic, output_dim))
        self.bs_segmentation = nn.Parameter(torch.zeros((output_dim,)))

        self.topic_classification = Topic_Pooling(output_dim=topic_size, input_dim=hidden_dim*2+args.num_topic)

        self.softmax = F.softmax
        self.top = 0.7

        nn.init.uniform_(self.Ws_segmentation, -0.1, 0.1)

    def pad(self, paper_list):
        len_paper_list = [paper.size(0) for paper in paper_list]
        return pad_sequence(paper_list, batch_first=True, padding_value=0), len_paper_list

    def forward(self, sentences_embedding_papers, lda_one_hot_papers, seg_label, enforced_teach=False):
        padded_sentences_embedding_papers, len_paper_list = self.pad(sentences_embedding_papers)

        padded_sentences_embedding_after_rnn, (_) = self.rnn(padded_sentences_embedding_papers)

        lda_topic_vectors_and_loss_papers = [self.lda(lda_one_hot_paper) for lda_one_hot_paper in lda_one_hot_papers]

        lda_topic_vectors_papers = [lda_topic_vectors_and_loss_paper[0]
                                    for lda_topic_vectors_and_loss_paper in lda_topic_vectors_and_loss_papers]
        lda_loss_papers = [torch.mean(lda_topic_vectors_and_loss_paper[1])
                           for lda_topic_vectors_and_loss_paper in lda_topic_vectors_and_loss_papers]

        # lda_loss = torch.mean(torch.tensor(lda_loss_papers, dtype=lda_loss_papers[0].dtype,
        #                                    device=lda_loss_papers[0].device, requires_grad=True))

        padded_lda_topic_vectors_papers, _ = self.pad(lda_topic_vectors_papers)

        padded_sentences_embedding = \
            torch.cat((padded_sentences_embedding_after_rnn, padded_lda_topic_vectors_papers), dim=-1)

        padded_segment_probability = self.softmax(padded_sentences_embedding.matmul(self.Ws_segmentation)
                                                  + self.bs_segmentation, dim=-1)
        padded_segment_label = \
            torch.ge(padded_segment_probability,
                     torch.full_like(padded_segment_probability, self.top)).long().chunk(2, -1)[0].squeeze(-1)
        padded_segment_probability = padded_segment_probability.squeeze(1).chunk(2, -1)[0].squeeze(-1)
        segment_probability_papers = [res_pro_tensor.squeeze(0)[:len_paper] for (res_pro_tensor, len_paper) in
                                      zip(padded_segment_probability.chunk(padded_segment_probability.size(0), 0), len_paper_list)]
        segment_label_papers = [segment_label_per_paper.squeeze(0)[:len_paper] for (segment_label_per_paper, len_paper)
                                in zip(padded_segment_label.chunk(padded_segment_label.size(0), 0), len_paper_list)]

        sentences_embedding_after_rnn_papers = \
            [padded_sentences_embedding_after_rnn_paper.squeeze(0)[:len_paper]
             for (padded_sentences_embedding_after_rnn_paper, len_paper)
                in zip(padded_sentences_embedding_after_rnn.chunk(padded_sentences_embedding_after_rnn.size(0), 0), len_paper_list)]

        sentences_embedding_papers = \
            [padded_sentences_embedding_paper.squeeze(0)[:len_paper]
             for (padded_sentences_embedding_paper, len_paper)
                in zip(padded_sentences_embedding.chunk(padded_sentences_embedding.size(0), 0), len_paper_list)]

        if not enforced_teach:
            topic_probability_papers, _ = self.topic_classification(segment_label_papers, sentences_embedding_papers, len_paper_list)
        else:
            topic_probability_papers, _ = self.topic_classification(seg_label, sentences_embedding_papers, len_paper_list)

        return segment_probability_papers, segment_label_papers, topic_probability_papers, lda_loss_papers


class Topic_Pooling(nn.Module):
    def __init__(self, output_dim, input_dim=400):
        super(Topic_Pooling, self).__init__()
        self.input_dim = input_dim
        self.pool_dim = 3 * input_dim
        self.output_dim = output_dim
        self.Ws = nn.Parameter(torch.Tensor(self.pool_dim, self.output_dim))
        self.bs = nn.Parameter(torch.zeros((self.output_dim,)))

        nn.init.uniform_(self.Ws, -0.1, 0.1)
        nn.init.uniform_(self.bs, -0.1, 0.1)

    def forward(self, seg_label, sentence_embedding, len_paper_list):
        papers_topic = []
        papers_topic_label = []
        for seg_label_per_paper, sentence_embedding_per_paper, paper_len in zip(seg_label, sentence_embedding,
                                                                                len_paper_list):
            seg_label_per_paper[0] = 1
            seg_label_per_paper = torch.cat(
                (seg_label_per_paper, torch.tensor([1], dtype=torch.long, device=sentence_embedding_per_paper.device)), -1)
            segment_location = torch.nonzero(seg_label_per_paper, as_tuple=False)
            segment_length = torch.zeros((segment_location.size(0) - 1, segment_location.size(1)), dtype=torch.long,
                                         device=sentence_embedding_per_paper.device)
            for i in range(1, segment_location.size(0)):
                segment_length[i - 1] = segment_location[i] - segment_location[i - 1]
            segment_length = segment_length.squeeze(-1).cpu().numpy().tolist()
            segment_embedding_split = torch.split(sentence_embedding_per_paper, segment_length, dim=0)

            def pooling(segment_embedding):
                mean_segment_embedding = torch.mean(segment_embedding, dim=0)
                max_segment_embedding = torch.max(segment_embedding, dim=0)[0]
                last_segment_embedding = segment_embedding[-1][:].clone()
                pool_segment_embedding = torch.cat(
                    (mean_segment_embedding, max_segment_embedding, last_segment_embedding)).unsqueeze(0)
                return pool_segment_embedding

            all_segment_embedding = torch.cat(
                [pooling(segment_embedding) for segment_embedding in segment_embedding_split], dim=0)
            topic_res = all_segment_embedding.matmul(self.Ws) + self.bs
            topic_res = topic_res.chunk(topic_res.size(0), 0)
            sentence_topic = torch.cat([per_topic.expand(topic_len, self.output_dim) for (per_topic, topic_len) in
                                        zip(topic_res, segment_length)], dim=0)
            papers_topic.append(sentence_topic)
            papers_topic_label.append(torch.argmax(sentence_topic, dim=-1))

        return papers_topic, papers_topic_label
