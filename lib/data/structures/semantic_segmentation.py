import cv2
import numpy as np
import os
from pycocotools import mask as maskUtils

import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class SemanticSegmentation(object):
    """
    This class handles semantic segmentation for all objects in the image
    """
    FLIP_MAP = ()
    def __init__(self, semseg, class_ids, size, mode='poly', extra_fields=None):
        self.size = size
        self.class_ids = class_ids
        self.mode = mode
        self.extra_fields = extra_fields.copy() if extra_fields is not None else extra_fields
        self.ignore_label = extra_fields['ignore_label'] if 'ignore_label' in extra_fields else 0
        self.label_shift = extra_fields['label_shift'] if 'label_shift' in extra_fields else 1

        semseg = convert_poly_to_semseg(size, semseg, class_ids, label_shift=self.label_shift, extra_fields=extra_fields) \
            if isinstance(semseg, list) and mode =='poly' else semseg
        if isinstance(semseg, torch.Tensor):
            # The raw data representation is passed as argument
            semseg = semseg.clone()
        elif isinstance(semseg, (list, tuple, np.ndarray)):
            semseg = torch.as_tensor(semseg)

        # single channel
        semseg = semseg.unsqueeze(0) if len(semseg.shape) == 2 else semseg
        assert len(semseg.shape) == 3 and semseg.shape[0] == 1

        self.semseg = semseg

        if 'pano_anns' in self.extra_fields:
            del self.extra_fields['pano_anns']

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")

        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_semseg = self.semseg.flip(dim)

        for l_r in SemanticSegmentation.FLIP_MAP:
            left = torch.where(flipped_semseg == l_r[0])
            right = torch.where(flipped_semseg == l_r[1])
            flipped_semseg[left] = l_r[1]
            flipped_semseg[right] = l_r[0]

        return SemanticSegmentation(flipped_semseg, self.class_ids, self.size, mode=self.mode, extra_fields=self.extra_fields)

    def move(self, gap):
        c, h, w = self.semseg.shape
        old_up, old_left, old_bottom, old_right = max(gap[1], 0), max(gap[0], 0), h, w
        new_up, new_left = max(0 - gap[1], 0), max(0 - gap[0], 0)
        new_bottom, new_right = h + new_up - old_up, w + new_left - old_left
        new_shape = (c, h + new_up, w + new_left)
        moved_semseg = torch.zeros(new_shape, dtype=torch.uint8) + self.ignore_label
        moved_semseg[:, new_up:new_bottom, new_left:new_right] = self.semseg[:, old_up:old_bottom, old_left:old_right]
        moved_size = new_shape[2], new_shape[1]

        return SemanticSegmentation(moved_semseg, self.class_ids, moved_size, mode=self.mode, extra_fields=self.extra_fields)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_semseg = self.semseg[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height

        return SemanticSegmentation(cropped_semseg, self.class_ids, cropped_size, mode=self.mode, extra_fields=self.extra_fields)

    def set_size(self, size):
        c, h, w = self.semseg.shape
        new_shape = (c, size[1], size[0])
        new_semseg = torch.zeros(new_shape, dtype=torch.uint8) + self.ignore_label
        new_semseg[:, :min(h, size[1]), :min(w, size[0])] = self.semseg[:, :min(h, size[1]), :min(w, size[0])]
        self.semseg = new_semseg

        return SemanticSegmentation(self.semseg, self.class_ids, size, mode=self.mode, extra_fields=self.extra_fields)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)
        # width, height = int(width * float(scale) + 0.5), int(height * float(scale) + 0.5)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_semseg = torch.nn.functional.interpolate(
            self.semseg[None].float(),
            size=(height, width),
            mode="nearest",
        )[0].type_as(self.semseg)

        resized_size = width, height
        return SemanticSegmentation(resized_semseg, self.class_ids, resized_size, mode=self.mode, extra_fields=self.extra_fields)

    def to(self, device):
        semseg = self.semseg.to(device)
        return SemanticSegmentation(semseg, self.class_ids, self.size, mode=self.mode, extra_fields=self.extra_fields)

    def get_semseg(self):
        return self.semseg

    def __getitem__(self, item):
        return SemanticSegmentation(self.semseg, self.class_ids, self.size, mode=self.mode, extra_fields=self.extra_fields)

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def __len__(self):
        return len(self.dp_uvs)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "semseg_shape={}, ".format(self.semseg.shape)
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s


def get_semseg(root_dir, name, extra_fields):
    """
    get picture form annotations when parsing task runs
    """
    seg_root = extra_fields['seg_root'] if 'seg_root' in extra_fields else root_dir.replace('img', 'seg')
    name_trans = extra_fields['name_trans'] if 'name_trans' in extra_fields else ['jpg', 'png']
    label_shift = extra_fields['label_shift'] if 'label_shift' in extra_fields else 0
    semseg_path = os.path.join(seg_root, name.replace(name_trans[0], name_trans[1]))
    if 'pano_anns' in extra_fields:
        return convert_pano_to_semseg(semseg_path, extra_fields, name.replace(name_trans[0], name_trans[1]))
    else:
        semseg = cv2.imread(semseg_path, 0) + label_shift
        return semseg.astype(np.uint8)


def semseg_batch_resize(tensors, size_divisible=0, scale=1 / 8, ignore_value=255):
    assert isinstance(tensors, list)
    if size_divisible > 0:
        max_size = tuple(max(s) for s in zip(*[semseg.shape for semseg in tensors]))
        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_semsegs = tensors[0].new(*batch_shape).zero_() + ignore_value
        for semseg, pad_semseg in zip(tensors, batched_semsegs):
            pad_semseg[: semseg.shape[0], : semseg.shape[1], : semseg.shape[2]].copy_(semseg)

        _, _, height, width = batched_semsegs.shape

        width, height = int(width * float(scale) + 0.5), int(height * float(scale) + 0.5)
        # Height comes first here!
        batched_resized_semsegs = torch.nn.functional.interpolate(
            batched_semsegs.float(),
            size=(height, width),
            mode="nearest",
        ).type_as(batched_semsegs)

        return batched_resized_semsegs


def convert_pano_to_semseg(image_path, extra_fileds, name):
    """Pre-process the panoptic annotations to semantic annotations,
    it can convert coco or cityscape format, but now, it only support
    coco format. It supports two mainstream process ways, get all
    thing's ids to one or not, which is using 'extra_field['convert_format']'
    to control.
    """
    from pet.lib.data.evaluation.panoptic_eval import rgb2id
    anns = extra_fileds['pano_anns']['annotations']
    ignore_label = extra_fileds['ignore_label']
    semseg_label_format = extra_fileds['semseg_label_format']
    assert semseg_label_format in ['thing_only', 'stuff_only', 'stuff_thing']
    categories = extra_fileds['pano_anns']['categories']
    thing_ids = [c['id'] for c in categories if c['isthing'] == 1]
    stuff_ids = [c['id'] for c in categories if c['isthing'] == 0]
    panoptic = rgb2id(cv2.imread(image_path)[..., ::-1])
    output = torch.ones(panoptic.shape, dtype=torch.uint8) * ignore_label

    if semseg_label_format == 'stuff_only':
        assert len(thing_ids) < ignore_label
        id_map = {stuff_id: i+1 for i, stuff_id in enumerate(stuff_ids)}
        thing_map = {thing_id: 0 for thing_id in thing_ids}
        id_map.update(thing_map)
    elif semseg_label_format == 'stuff_thing':
        pan_ids = stuff_ids + thing_ids
        assert len(pan_ids) < ignore_label
        id_map = {pan_id: i for i, pan_id in enumerate(pan_ids)}
    else:
        raise NotImplementedError
    id_map.update({0: ignore_label})
    # TODO find a ideal way to get specific annotation
    for ann in anns:
        if ann['file_name'] == name:
            segments = ann['segments_info']
            break

    for seg in segments:
        cat_id = seg["category_id"]
        new_cat_id = id_map[cat_id]
        output[panoptic == seg["id"]] = new_cat_id

    return output


def convert_poly_to_semseg(size, semsegs_anno, things_classes, label_shift=1, extra_fields=None):
    w, h = size
    assert 'semseg_label_format' in extra_fields
    semseg_label_format = extra_fields['semseg_label_format']
    assert semseg_label_format in ['thing_only', 'stuff_only', 'stuff_thing']

    def paste_poly_anno(img, annos, class_ids):
        if len(annos) and len(class_ids):
            for semseg_anno_per_class, class_id in zip(annos, class_ids):
                if type(semseg_anno_per_class) == list:
                    rles = maskUtils.frPyObjects(semseg_anno_per_class, h, w)
                    rle = maskUtils.merge(rles)
                elif type(semseg_anno_per_class['counts']) == list:
                    rle = maskUtils.frPyObjects(semseg_anno_per_class, h, w)
                else:
                    rle = semseg_anno_per_class
                m = maskUtils.decode(rle)
                img[m == 1] = class_id

    things_seg, stuff_annos = semsegs_anno
    stuff_seg = []
    stuff_classes = []
    init_label = 0
    if stuff_annos is not None and semseg_label_format != 'thing_only':
        assert 'json_category_id' in extra_fields
        assert 'json_category_id_to_contiguous_id' in extra_fields
        seg_class = extra_fields["json_category_id"]
        seg_class_dict = extra_fields["json_category_id_to_contiguous_id"]
        new_things_classes = things_classes.clone().tolist()
        init_label = seg_class_dict[seg_class[-1]]

        if semseg_label_format == 'stuff_only':
            new_things_classes = [0 for _ in new_things_classes]
            seg_class_dict = {v: i for i, v in enumerate(seg_class)}
            init_label = seg_class_dict[seg_class[-1]] + 1

        stuff_classes = [obj["category_id"] for obj in stuff_annos]
        stuff_classes = torch.tensor([seg_class_dict[c] for c in stuff_classes])
        stuff_seg = [obj["segmentation"] for obj in stuff_annos]

        if semseg_label_format == 'stuff_only':
            stuff_classes = stuff_classes + label_shift
        stuff_classes = stuff_classes.tolist()
    else:
        assert semseg_label_format == 'thing_only'
        new_things_classes = (things_classes + label_shift).tolist()

    img = np.ones((h, w), dtype=np.uint8) * init_label
    paste_poly_anno(img, stuff_seg, stuff_classes)
    paste_poly_anno(img, things_seg, new_things_classes)
    return torch.from_numpy(img)
