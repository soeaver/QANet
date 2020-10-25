import torch

from lib.data.structures.instance import get_affine_transform


class InstanceBox(object):
    def __init__(self, bbox, labels, image_size, scores=None, ori_bbox=None, im_bbox=None):
        device = bbox.device if isinstance(bbox, torch.Tensor) else torch.device("cpu")
        bbox = torch.as_tensor(bbox, dtype=torch.float32, device=device)
        labels = torch.as_tensor(labels)
        scores = torch.as_tensor(scores) if scores is not None else torch.ones(len(bbox))
        ori_bbox = torch.as_tensor(ori_bbox) if ori_bbox is not None else bbox
        if im_bbox is not None:
            im_bbox = torch.as_tensor(im_bbox)
        else:
            xmin, ymin, w, h = bbox.split(1, dim=-1) # xywh t0 xyxy
            im_bbox = torch.cat(
                (xmin, ymin, xmin + w, ymin + h), dim=-1
            )
        if bbox.ndimension() != 2:
            raise ValueError("bbox should have 2 dimensions, got {}".format(bbox.ndimension()))
        if bbox.size(-1) != 4 and bbox.size(-1) != 5:
            raise ValueError("last dimension of bbox should have a size of 4 or 5, got {}".format(bbox.size(-1)))

        self.bbox = bbox
        self.labels = labels
        self.scores = scores
        self.ori_bbox = ori_bbox
        self.im_bbox = im_bbox
        self.size = image_size # (w, h)
        self.extra_fields = {}
        self.trans = None

    def convert(self, aspect_ratio, scale_ratio):
        bbox = self.bbox
        bbox[:, 0] += bbox[:, 2] / 2.0
        bbox[:, 1] += bbox[:, 3] / 2.0
        xc, yc, w, h = bbox.split(1, dim=-1)
        h[w > aspect_ratio * h] = w[w > aspect_ratio * h] * 1.0 / aspect_ratio
        w[w < aspect_ratio * h] = h[w < aspect_ratio * h] * aspect_ratio
        w *= 1.25
        h *= 1.25
        rot = torch.zeros(xc.shape).to(dtype=xc.dtype)
        self.ori_bbox = torch.cat((xc, yc, w, h, rot), dim=-1)
        self.bbox = self.ori_bbox * scale_ratio

    def crop_and_resize(self, train_size, affine_mode='cv2'):
        if affine_mode == 'cv2':
            self.trans = []
            for box in self.bbox:
                self.trans.append(get_affine_transform(box, train_size))

    def __len__(self):
        return self.bbox.shape[0]

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_boxes={}, ".format(len(self))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


def instancebox_split(instancebox, batch_size):
    bbox = instancebox.bbox.split(batch_size, dim=0)
    labels = instancebox.labels.split(batch_size, dim=0)
    scores = instancebox.scores.split(batch_size, dim=0)
    ori_bbox = instancebox.ori_bbox.split(batch_size, dim=0)
    im_bbox = instancebox.im_bbox.split(batch_size, dim=0)
    image_size = instancebox.size
    results = [InstanceBox(_bbox, _labels, image_size, _scores, _ori_bbox, _im_bbox)
               for _bbox, _labels, _scores, _ori_bbox, _im_bbox
               in zip(bbox, labels, scores, ori_bbox, im_bbox)]
    return results
