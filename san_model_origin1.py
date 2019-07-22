import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

# concatenate version, match dimension later
class SANModel1(nn.Module):
    def __init__(self, w_emb, q_emb, v_att1, v_att2, q_net, v_net, classifier,):
        super(SANModel1, self).__init__()
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
        q_emb = self.q_emb(w_emb) # [batch, q_dim]

        # The first attention layer
        att = self.v_att1(v, q_emb) # [batch, k, 1]
        v_emb = (att * v).sum(1) # [batch, v_dim]

        # The second attention layer
        q_new_2 = torch.cat((q_emb, v_emb), 1) # [batch, q_dim + v_dim]
        att2 = self.v_att2(v, q_new_2)
        v_emb2 = (att2 * v).sum(1) # [batch, v_dim]

        q_new_3 = torch.cat((q_emb, v_emb2), 1)  # [batch, q_dim + v_dim]
        att3 = self.v_att2(v, q_new_3)
        v_emb3 = (att3 * v).sum(1)  # [batch, v_dim]

        # q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb3) # [batch, num_hid]
        # joint_repr = q_repr * v_repr

        logits = self.classifier(v_repr)
        # print(logits.shape)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att1 = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    v_att2 = Attention(dataset.v_dim, q_emb.num_hid + dataset.v_dim, num_hid)

    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return SANModel1(w_emb, q_emb, v_att1, v_att2, q_net, v_net, classifier)

