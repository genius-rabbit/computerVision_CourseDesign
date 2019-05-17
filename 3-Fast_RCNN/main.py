from data_utils import data_load
from fast_rcnn import FAST_RCNN
from torch.autograd import Variable
import torch
import numpy as np

use_cuda = torch.cuda.is_available()

# load and random data set
print('load data set...')
x_tr, y_tr, box_tr, x_te, y_te, box_te = data_load()

# build model

print('build model...')
if use_cuda:
    rcnn = FAST_RCNN().cuda()
else:
    rcnn = FAST_RCNN()
print(rcnn)
optimizer = torch.optim.Adam(rcnn.parameters(), lr=1e-4)


def train_batch(img, rois, ridx, gt_cls, gt_tbbox, test_flag=False):
    sc, r_bbox = rcnn(img, rois, ridx)
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)
    fl = loss.data.cpu().numpy()[0]
    fl_sc = loss_sc.data.cpu().numpy()[0]
    fl_loc = loss_loc.data.cpu().numpy()[0]

    if not test_flag:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc


def train_epoch(data_set, test_flag=False):
    batch_img_nums = 10

    img_nums = data_set.shape[0]
    perm = np.random.permutation(img_nums)
    data_set = data_set[perm]

    losses = []
    losses_sc = []
    losses_loc = []
    for i in range(0, img_nums, batch_img_nums):
        batch_l = i
        batch_r = min(i + batch_img_nums, img_nums)

        one_img_batch = Variable(torch.from_numpy(data_set[batch_l:batch_r]), volatile=test_flag)  # .cuda()

        ridx = []
        glo_ids = []

        glo_ids = np.concatenate(glo_ids, axis=0)
        rois = box_tr[glo_ids]
        print(torch.from_numpy(y_tr[glo_ids]).shape)
        gt_cls = Variable(torch.from_numpy(y_tr[glo_ids]), volatile=test_flag)  # .cuda()
        gt_tbbox = Variable(torch.from_numpy(box_tr[glo_ids]), volatile=test_flag)  # .cuda()
        print(gt_cls)

        loss, loss_sc, loss_loc = train_batch(one_img_batch, rois, ridx, gt_cls, gt_tbbox, test_flag=test_flag)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    print(f'Avg loss = {avg_loss:.4f}; loss_sc = {avg_loss_sc:.4f}, loss_loc = {avg_loss_loc:.4f}')


train_epoch(x_tr, False)
exit()