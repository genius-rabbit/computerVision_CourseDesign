from data_utils import *
from fast_rcnn import FAST_RCNN
from torch.autograd import Variable
from roi import *
import pickle

use_cuda = torch.cuda.is_available()
N_CLASS = 4

# load and random data set
print('load data set...')
x_tr, y_tr, box_tr, x_te, y_te, box_te = data_load()

# build model

print('build model...')
rcnn = FAST_RCNN()
print(rcnn)
optimizer = torch.optim.Adam(rcnn.parameters(), lr=1e-4)


def train_batch(img, rois, gt_cls, gt_tbbox, test_flag=False):
    sc, r_bbox = rcnn(img, rois)
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)
    fl = loss.item()
    fl_sc = loss_sc.item()
    fl_loc = loss_loc.item()

    if not test_flag:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return fl, fl_sc, fl_loc


def train_epoch(data_set, test_flag=False):
    batch_img_nums = 50
    img_nums = data_set.shape[0]

    losses = []
    losses_sc = []
    losses_loc = []
    for i in range(0, img_nums, batch_img_nums):
        batch_l = i
        batch_r = min(i + batch_img_nums, img_nums)

        one_img_batch = Variable(torch.from_numpy(data_set[batch_l:batch_r]).float(), volatile=test_flag)  # .cuda()

        gt_cls = torch.from_numpy(y_tr[batch_l:batch_r]).long()
        gt_tbbox = torch.from_numpy(box_tr[batch_l:batch_r]).float()
        # print(gt_cls.shape, gt_tbbox.shape)
        gt_cls = Variable(gt_cls, volatile=test_flag)
        gt_tbbox = Variable(gt_tbbox, volatile=test_flag)

        rois = box_tr[batch_l:batch_r]
        loss, loss_sc, loss_loc = train_batch(one_img_batch, rois, gt_cls, gt_tbbox, test_flag=test_flag)
        losses.append(loss)
        losses_sc.append(loss_sc)
        losses_loc.append(loss_loc)

        print('batch = %3d - %3d loss = %.6f loss_sc = %.6f loss_loc = %.6f' % (i, i + batch_img_nums, loss, loss_sc, loss_loc))

    avg_loss = np.mean(losses)
    avg_loss_sc = np.mean(losses_sc)
    avg_loss_loc = np.mean(losses_loc)
    print('Avg loss = %.6f loss_sc = %.6f loss_loc = %.6f ' % (avg_loss.item(), avg_loss_sc.item(), avg_loss_loc.item()))


def test(imgs, label, box):
    # input img
    box.tofile('old_box') # float64
    # print(box.dtype)
    imgs.tofile('imgs') # uint8
    # print(imgs.dtype)

    img_batch = Variable(torch.from_numpy(imgs[:]).float())
    sc, r_bbox = rcnn(img_batch, box)

    # acc
    probs = sc.detach().numpy()
    pred = np.argmax(probs, axis=1)
    acc = np.sum(pred == label) / float(label.shape[0])

    # loss
    gt_cls = Variable(torch.from_numpy(label[:]).long())
    gt_tbbox = Variable(torch.from_numpy(box[:]).float())
    loss, loss_sc, loss_loc = rcnn.calc_loss(sc, r_bbox, gt_cls, gt_tbbox)

    # IOU avg
    lbl = gt_cls.view(-1, 1, 1).expand(gt_cls.size(0), 1, 4)
    mask = (gt_cls != 0).float().view(-1, 1).expand(gt_cls.size(0), 4)
    res_bbox = r_bbox.gather(1, lbl).squeeze(1) * mask
    res_bbox = res_bbox.detach().numpy()

    res_bbox.tofile('box') # float32
    # print(res_bbox.dtype)
     
    iou_avg = calc_ious(res_bbox, box)

    print('loss: %.6f loss_sc:%.6f loss_loc:%.6f iou_avg:%.6f acc:%.6f' % (loss.item(), loss_sc.item(), loss_loc.item(), iou_avg, acc))


def train_test():
    for i in range(100):
        train_epoch(x_tr)
        test(x_te, y_te, box_te)


print('Start train...')
train_test()
exit(0)