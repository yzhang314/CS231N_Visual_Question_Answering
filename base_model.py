import torch
import torch.nn as nn
from attention import Attention, NewAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet

# LSTM model
class BaseModel(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, q_net, v_net, classifier, lstm=None, w_emb2=None):
        super(BaseModel, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        self.w_emb2 = w_emb2
        self.lstm = lstm
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

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        # q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb) # [batch, num_hid]
        w_emb2 = self.w_emb2(q)
        # joint_repr = q_repr * v_repr
        embeddings = torch.cat((v_repr.unsqueeze(1), w_emb2), 1) # [batch, seq + 1, 300]
        output, (hidden, cell) = self.lstm(embeddings) # hidden -> [1, batch, hid_dim]
        # logits = self.classifier(joint_repr)
        logits = self.classifier(hidden.squeeze(0))
        # print(logits.shape)
        return logits


def build_baseline0(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)


def build_baseline0_newatt(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = NewAttention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([q_emb.num_hid, num_hid])
    v_net = FCNet([dataset.v_dim, num_hid])
    classifier = SimpleClassifier(
        num_hid, num_hid * 2, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier)

def build_baseline1(dataset, num_hid):
        w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
        w_emb2 = WordEmbedding(dataset.dictionary.ntoken, num_hid, 0.0)
        q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
        v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
        q_net = FCNet([num_hid, num_hid])
        v_net = FCNet([dataset.v_dim, num_hid])
        lstm = nn.LSTM(num_hid, num_hid, 1, batch_first=True)
        classifier = SimpleClassifier(num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
        return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, lstm, w_emb2)

def build_baseline2(dataset, num_hid):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0)
    q_emb = QuestionEmbedding(300, num_hid, 1, False, 0.0)
    v_att = Attention(dataset.v_dim, q_emb.num_hid, num_hid)
    q_net = FCNet([num_hid, num_hid])
    v_net = nn.Linear(dataset.v_dim, 300)
    v_bn = nn.BatchNorm1d(300, momentum=0.01)
    lstm = nn.LSTM(300, num_hid, 1, batch_first=True)
    classifier = SimpleClassifier(num_hid, 2 * num_hid, dataset.num_ans_candidates, 0.5)
    return BaseModel(w_emb, q_emb, v_att, q_net, v_net, classifier, lstm, v_bn)

