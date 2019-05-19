import torchvision
from roi import *
import torch.nn as nn
from torch.autograd import Variable

N_CLASS = 4


class FAST_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # use vgg 16 bn <bn use batch norm layer>
        raw_net = torchvision.models.vgg16_bn(pretrained=True).cpu()
        # conv layer sequence
        self.seq = nn.Sequential(*list(raw_net.features.children())[:-1])
        self.roi_pool = SlowROIPool(output_size=(7, 7))

        # classifier layer sequence
        self.feature = nn.Sequential(*list(raw_net.classifier.children())[:-1])
        # print(self.feature)

        _x = Variable(torch.Tensor(1, 3, 128, 128))
        _r = np.array([[0., 0., 1., 1.]])
        # _ri = np.array([0])
        _x = self.feature(self.roi_pool(self.seq(_x), _r).view(1, -1))
        feature_dim = _x.size(1)
        self.cls_score = nn.Linear(feature_dim, N_CLASS + 1)
        self.bbox = nn.Linear(feature_dim, 4 * (N_CLASS + 1))

        self.cel = nn.CrossEntropyLoss()
        self.sl1 = nn.SmoothL1Loss()

    def forward(self, imgs, rois):
        res = self.seq(imgs)
        res = self.roi_pool(res, rois)
        res = res.detach()
        res = res.view(res.size(0), -1)
        feat = self.feature(res)

        cls_score = self.cls_score(feat)
        bbox = self.bbox(feat).view(-1, N_CLASS + 1, 4)
        return cls_score, bbox

    def calc_loss(self, probs, bbox, labels, gt_bbox):
        loss_sc = self.cel(probs, labels)
        lbl = labels.view(-1, 1, 1).expand(labels.size(0), 1, 4)
        mask = (labels != 0).float().view(-1, 1).expand(labels.size(0), 4)

        loss_loc = self.sl1(bbox.gather(1, lbl).squeeze(1) * mask, gt_bbox * mask)
        lmb = 1.0
        loss = loss_sc + lmb * loss_loc
        return loss, loss_sc, loss_loc
