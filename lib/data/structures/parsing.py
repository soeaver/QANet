import cv2
import os

import torch

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class Parsing(object):
    """
    This class handles parsing for all objects in the image
    """
    FLIP_MAP = ()

    def __init__(self, parsing, size):
        if isinstance(parsing, torch.Tensor):
            # The raw data representation is passed as argument
            parsing = parsing.clone()
        elif isinstance(parsing, (list, tuple)):
            parsing = torch.as_tensor(parsing)

        if len(parsing.shape) == 2:
            # if only a single instance mask is passed
            parsing = parsing[None]

        assert len(parsing.shape) == 3
        assert parsing.shape[1] == size[1], "%s != %s" % (parsing.shape[1], size[1])
        assert parsing.shape[2] == size[0], "%s != %s" % (parsing.shape[2], size[0])

        self.parsing = parsing
        self.size = size
        self.extra_fields = {}

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")

        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_parsing = self.parsing.flip(dim)

        for l_r in Parsing.FLIP_MAP:
            left = torch.where(flipped_parsing == l_r[0])
            right = torch.where(flipped_parsing == l_r[1])
            flipped_parsing[left] = l_r[1]
            flipped_parsing[right] = l_r[0]

        return Parsing(flipped_parsing, self.size)

    def move(self, gap):
        c, h, w = self.parsing.shape
        old_up, old_left, old_bottom, old_right = max(gap[1], 0), max(gap[0], 0), h, w

        new_up, new_left = max(0 - gap[1], 0), max(0 - gap[0], 0)
        new_bottom, new_right = h + new_up - old_up, w + new_left - old_left
        new_shape = (c, h + new_up, w + new_left)

        moved_parsing = torch.zeros(new_shape, dtype=torch.uint8)
        moved_parsing[:, new_up:new_bottom, new_left:new_right] = self.parsing[:, old_up:old_bottom, old_left:old_right]

        moved_size = new_shape[2], new_shape[1]
        return Parsing(moved_parsing, moved_size)

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
        cropped_parsing = self.parsing[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return Parsing(cropped_parsing, cropped_size)

    def set_size(self, size):
        c, h, w = self.parsing.shape
        new_shape = (c, size[1], size[0])

        new_parsing = torch.zeros(new_shape, dtype=torch.uint8)
        new_parsing[:, :min(h, size[1]), :min(w, size[0])] = self.parsing[:, :min(h, size[1]), :min(w, size[0])]

        self.parsing = new_parsing
        return Parsing(self.parsing, size)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_parsing = torch.nn.functional.interpolate(
            self.parsing[None].float(),
            size=(height, width),
            mode="nearest",
        )[0].type_as(self.parsing)

        resized_size = width, height
        return Parsing(resized_parsing, resized_size)

    def to(self, *args, **kwargs):
        return self

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def __len__(self):
        return len(self.parsing)

    def __getitem__(self, index):
        parsing = self.parsing[index].clone()
        return Parsing(parsing, self.size)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_parsing = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_parsing
        raise StopIteration()

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_parsing={}, ".format(len(self.parsing))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


def get_parsing(root_dir, file_name, parsing_ids):
    human_dir = root_dir.replace('Images', 'Human_ids')
    category_dir = root_dir.replace('Images', 'Category_ids')
    file_name = file_name.replace('jpg', 'png')
    human_path = os.path.join(human_dir, file_name)
    category_path = os.path.join(category_dir, file_name)
    human_mask = cv2.imread(human_path, 0)
    category_mask = cv2.imread(category_path, 0)
    parsing = []
    for id in parsing_ids:
        parsing.append(category_mask * (human_mask == id))
    return parsing


def parsing_on_boxes(parsing, rois, prob_size):
    device = rois.device
    rois = rois.to(torch.device("cpu"))
    parsing_list = []
    for i in range(rois.shape[0]):
        parsing_ins = parsing[i].cpu().numpy()
        xmin, ymin, xmax, ymax = torch.round(rois[i]).int()
        cropped_parsing = parsing_ins[ymin:ymax, xmin:xmax]
        resized_parsing = cv2.resize(cropped_parsing, (prob_size[1], prob_size[0]), interpolation=cv2.INTER_NEAREST)
        parsing_list.append(torch.from_numpy(resized_parsing))

    if len(parsing_list) == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.stack(parsing_list, dim=0).to(device, dtype=torch.int64)


def flip_parsing_prob(parsing_prob, flip_map):
    parsing_prob_flipped = torch.tensor(parsing_prob)
    for i in flip_map:
        parsing_prob_flipped[:, i[0], :, :] = parsing_prob[:, i[1], :, :]
        parsing_prob_flipped[:, i[1], :, :] = parsing_prob[:, i[0], :, :]
    parsing_prob_flipped = torch.flip(parsing_prob_flipped, dims=(3,))

    return parsing_prob_flipped
