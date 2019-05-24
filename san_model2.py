import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

# Match dimension first
class SANModel2(nn.Module):
    def __init__(self, w_emb, q_emb, v_att1, v_att2, q_net, v_net, classifier,):
        super(SANModel2, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att1 = v_att1
        self.v_att2 = v_att2
        self.q_net = q_net
        self.v_net = v_net
        self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q) # [batch, seq, 300]
        q_emb = self.q_emb(w_emb) # [batch, num_hid]
        q_emb = self.q_net(q_emb) # [batch, num_hid]

        v_emb = self.v_net(v) # [batch, num_hid]

        # The first attention layer

        att = self.v_att1(v_emb, q_emb) # [batch, k, 1]
        v_emb = (att * v_emb).sum(1) # [batch, num_hid]


        q_new = q_emb + v_emb

        # The second attention layer
        att2 = self.v_att2(v, q_new)
        v_emb2 = (att2 * v).sum(1) # [batch, num_hid]

        joint_repr = q_new + v_emb2

        logits = self.classifier(joint_repr)
        # print(logits.shape)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att1 = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    v_att2 = Attention(dataset.v_dim, num_hid, num_hid)

    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return SANModel2(w_emb, q_emb, v_att1, v_att2, q_net, v_net, classifier)

