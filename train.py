import os
import time
import torch
import torch.nn as nn
import utils
from torch.autograd import Variable
import numpy as np


def instance_bce_with_logits(logits, labels):
    assert logits.dim() == 2

    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    loss *= labels.size(1)
    return loss


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argma
    # print (logits.size())
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


def train(model, train_loader, eval_loader, num_epochs, output):
    utils.create_dir(output)
    optim = torch.optim.Adamax(model.parameters())
    logger = utils.Logger(os.path.join(output, 'log.txt'))
    best_eval_score = 0
    f = open("cnn_rnn_best_2.txt", "w+")
    f.write('this is cnn rnn best model')


    for epoch in range(num_epochs):
        total_loss = 0
        train_score = 0
        t = time.time()

        for i, (v, b, q, a, q_id) in enumerate(train_loader):
            v = Variable(v).cuda()
            b = Variable(b).cuda()
            q = Variable(q).cuda()
            a = Variable(a).cuda()
            # print q_id.size()
            pred = model(v,b,q,a)
            """
            if (i == 0):
                q_id_new = q_id.view(-1, 1) #[512, ]
               # print (q_id_new.size())
                pred_label = torch.max(pred, 1)[1].data.cpu().view(-1, 1) # [512, ]
               #  print (pred_label.size())
               # result = torch.cat((q_id_new, pred_label), dim=1)
               
               # f1 = open("q_id", "ab")
               # f.write(q_id_new)
               # f1.close()
                np.savetxt('q_id', q_id_new.numpy(), fmt="%d")
                np.savetxt('label_prediction', pred_label.numpy(), fmt="%d")
            """
            #pred = model(v, b, q, a)
            loss = instance_bce_with_logits(pred, a)
            loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.25)
            optim.step()
            optim.zero_grad()

            batch_score = compute_score_with_logits(pred, a.data).sum()
            total_loss += torch.Tensor.item(loss.data) * v.size(0)
            train_score += batch_score

        total_loss /= len(train_loader.dataset)
        train_score = 100 * train_score / len(train_loader.dataset)
        model.train(False)
        eval_score, bound = evaluate(model, eval_loader, epoch)
        model.train(True)
        logger.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        logger.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        logger.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
        f.write('epoch %d, time: %.2f' % (epoch, time.time() - t))
        f.write('\ttrain_loss: %.2f, score: %.2f' % (total_loss, train_score))
        f.write('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))

        if eval_score > best_eval_score:
            model_path = os.path.join(output, 'model.pth')
            torch.save(model.state_dict(), model_path)
            best_eval_score = eval_score
    f.close()


def evaluate(model, dataloader,epoch):
    score = 0
    upper_bound = 0
    num_data = 0
    i = 1
    for v, b, q, a, q_id in iter(dataloader):
        v = Variable(v, volatile=True).cuda()
        b = Variable(b, volatile=True).cuda()
        q = Variable(q, volatile=True).cuda()
        pred = model(v, b, q, None)
        # write the prediction result to txt file
        if (epoch == 13):
            q_id_current = q_id.view(-1, 1).numpy()
            pred_label_current = torch.max(pred, 1)[1].data.cpu().view(-1,1).numpy()
            if (i == 1):
                q_id_new = q_id_current
                pred_label = pred_label_current
            else:
                q_id_new = np.concatenate((q_id_new, q_id_current), axis = 0)
                pred_label = np.concatenate((pred_label, pred_label_current), axis = 0)
            i = i+1
        batch_score = compute_score_with_logits(pred, a.cuda()).sum()
        score += batch_score
        upper_bound += (a.max(1)[0]).sum()
        num_data += pred.size(0)
    if (epoch == 13):
        np.savetxt('q_id_result_2', q_id_new, fmt="%d")
        np.savetxt('label_prediction_result_2', pred_label, fmt="%d")
    score = score / len(dataloader.dataset)
    upper_bound = upper_bound / len(dataloader.dataset)
    return score, upper_bound
