import torch
import torch.nn as nn
from attention import Attention, NewAttention, StackAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet


class SANModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, linear):
        super(SANModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier
        self.linear = linear

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q) # [batch, seq, 300]
        q_emb = self.q_emb(w_emb) # [batch, q_dim/num_hid]

        v_emb = self.linear(v) # [batch, k, num_hid]

        # q_repr = self.q_net(q_emb) # [batch, num_hid]
        # v_repr = self.v_net(v.sum(1))  # [batch, num_hid]

        # The SAN attention
        att_output = self.v_att(v_emb, q_emb) # [batch, hidden]

        logits = self.classifier(att_output)
        # print(logits.shape)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = StackAttention(num_hid, num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    linear = torch.nn.Linear(dataset.v_dim, num_hid)
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return SANModel(w_emb, q_emb, v_att, q_net, v_net, classifier, linear)

