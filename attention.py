import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from fc import FCNet


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([v_dim + q_dim, num_hid])
        self.linear = weight_norm(nn.Linear(num_hid, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1) # [batch, k, qdim]
        vq = torch.cat((v, q), 2) # [batch, k, vdim + qdim]
        joint_repr = self.nonlinear(vq) # [batch, k, num_hid]
        logits = self.linear(joint_repr)# [batch, k, 1]
        return logits

class NewAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid, dropout=0.2):
        super(NewAttention, self).__init__()

        self.v_proj = FCNet([v_dim, num_hid])
        self.q_proj = FCNet([q_dim, num_hid])
        self.dropout = nn.Dropout(dropout)
        self.linear = weight_norm(nn.Linear(q_dim, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, num_hid]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits

class StackAttention(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(StackAttention, self).__init__()

        # in this case we have q_dim = v_dim = num_hidden = 1024
        self.input_size = v_dim
        self.fc_q1 = nn.Linear(q_dim, 768, bias=True)
        self.fc_q2 = nn.Linear(768, 640, bias=True)
        self.fc_v1 = nn.Linear(v_dim, 768, bias=False)
        self.fc_v2 = nn.Linear(768, 640, bias=False)
        self.att_size = 512
        self.linear1 = nn.Linear(640, self.att_size, bias=False)
        self.fc_vq1 = nn.Linear(self.att_size, 1, bias=True)
        self.sf = nn.Softmax()

        self.fc_q3 = nn.Linear(self.input_size, self.att_size, bias=True)
        self.fc_v3 = nn.Linear(self.input_size, self.att_size, bias=False)
        self.fc_vq2 = nn.Linear(self.att_size, 1, bias=True)

        self.fc = nn.Linear(self.input_size, num_hid, bias=True)
        self.dp = nn.Dropout(0.5)

    def forward(self, v, q):
        """
            v: [batch, k, hidden]
            q: [batch, hidden]
        """
        B = q.size(0)
        k = v.size(1)

        # stack 1
        ques_emb = self.fc_q2(self.fc_q1(q))  # [batch, 640]
        img_emb = self.fc_v2(self.fc_v1(v))  # [batch, k, 640]

        h1 = self.tan(ques_emb.view(B, 1, 640) + img_emb)  # [batch, k, 640]
        h1_emb = self.linear1(h1)  # [batch, k, att_size]
        h1_emb = self.fc_vq1(self.dp(h1_emb))  # [batch, k, 1]
        p1 = self.sf(h1_emb.view(-1, k)).view(B, 1, k)  # [batch, 1, k]

        # weighted sum
        img_att1 = p1.matmul(v)  # [batch, 1, hidden]
        u1 = q + img_att1.view(-1, self.v_dim)  # [batch, hidden]

        # stack 2
        ques_emb2 = self.fc_q3(u1)  # [batch, att_size]
        img_emb2 = self.fc_v3(v)  # [batch, k, att_size]

        h2 = self.tan(ques_emb2.view(B, 1, self.att_size) + img_emb2)  # [batch, k, att_size]
        h2_emb = self.fc_vq2(self.dp(h2))  # [batch, k, 1]
        p2 = self.sf(h2_emb.view(-1, k)).view(B, 1, k)  # [batch, 1, k]

        # weighted sum
        img_att2 = p2.matmul(v)  # [batch, 1, hidden]
        u2 = u1 + img_att2.view(-1, self.v_dim)  # [batch, hidden]

        return u2
    # pass u2 to classifier later

class StackAttention1(nn.Module):
    def __init__(self, v_dim, q_dim, num_hid):
        super(StackAttention1, self).__init__()
        # in this case we have q_dim = v_dim = num_hidden = 1024
        self.input_size = v_dim
        self.fc_q1 = FCNet([q_dim, 768])
        self.fc_q2 = FCNet([768, 640])
        self.fc_v1 = FCNet([v_dim, 768])
        self.fc_v2 = FCNet([768, 640])
        self.att_size = 512
        self.linear1 = FCNet([640, self.att_size])
        self.fc_vq1 = FCNet([self.att_size, 1])
        self.tan = nn.Tanh()
        self.dp = nn.Dropout(0.5)

    def forward(self, v, q):
        """
        v: [batch, k, num_hid]
        q: [batch, num_hid]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        B = q.size(0)
        k = v.size(1)

        # stack 1
        ques_emb = self.fc_q2(self.fc_q1(q))  # [batch, 640]
        img_emb = self.fc_v2(self.fc_v1(v))  # [batch, k, 640]
        h1 = self.tan(ques_emb.view(B, 1, 640) + img_emb)  # [batch, k, 640]
        h1_emb = self.linear1(h1)  # [batch, k, att_size]
        logits = self.fc_vq1(self.dp(h1_emb))  # [batch, k, 1]

        return logits

class MUTAN(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers):
        super(MUTAN, self).__init__()
        # in this case we have q_dim = v_dim = num_hidden = 1024
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.num_layers = num_layers

        hv = []
        for i in range(self.num_layers):
            dp = nn.Dropout(p=0.1)
            linear = nn.Linear(self.input_dim, self.output_dim)
            tan = nn.Tanh()
            hv.append(nn.Sequential(dp, linear, tan))

        self.v_layer = nn.ModuleList(hv)

        hq = []
        for i in range(self.num_layers):
            dp = nn.Dropout(p=0.1)
            linear = nn.Linear(self.input_dim, self.output_dim)
            tan = nn.Tanh()
            hq.append(nn.Sequential(dp, linear, tan))

        self.q_layer = nn.ModuleList(hq)

        self.tan = nn.Tanh()


    def forward(self, v, q):
        """
        v: [batch, num_hid]
        q: [batch, num_hid]
        """
        batch_size = v.size()[0]
        x_mm = []
        for i in range(self.num_layers):
            x_hv = v
            x_hv = self.v_layer[i](x_hv)

            x_hq = q
            x_hq = self.q_layer[i](x_hq)
            x_mm.append(torch.mul(x_hq, x_hv))
        x_mm = torch.stack(x_mm, dim=1)
        x_mm = x_mm.sum(1).view(batch_size, self.output_dim)
        x_mm = self.tan(x_mm)
        return x_mm


