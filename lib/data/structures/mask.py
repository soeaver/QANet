import copy
import cv2
import numpy as np
import pycocotools.mask as mask_utils

import torch

from lib.ops import roi_align

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1
TO_REMOVE = 0

""" ABSTRACT
Masks come in either:
1) Binary masks
2) Polygons

Binary masks can be represented in a contiguous array
and operations can be carried out more efficiently,
therefore BinaryMaskList handles them together.

Polygons are handled separately for each instance,
by PolygonInstance and instances are handled by
PolygonList.

Mask is supposed to represent both,
therefore it wraps the functions of BinaryMaskList
and PolygonList to make it transparent.
"""


class BinaryMaskList(object):
    """
    This class handles binary masks for all objects in the image
    """

    def __init__(self, masks, size):
        """
            Arguments:
                masks: Either torch.tensor of [num_instances, H, W]
                    or list of torch.tensors of [H, W] with num_instances elems,
                    or RLE (Run Length Encoding) - interpreted as list of dicts,
                    or BinaryMaskList.
                size: absolute image size, width first

            After initialization, a hard copy will be made, to leave the
            initializing source data intact.
        """

        if isinstance(masks, torch.Tensor):
            # The raw data representation is passed as argument
            masks = masks.clone()
        elif isinstance(masks, (list, tuple)):
            if isinstance(masks[0], torch.Tensor):
                masks = torch.stack(masks, dim=2).clone()
            elif isinstance(masks[0], dict) and "counts" in masks[0]:
                # RLE interpretation,  in RLE, height come first in "size"
                assert all([(size[1], size[0]) == tuple(inst["size"]) for inst in masks])
                masks = mask_utils.decode(masks)  # [h, w, n]
                masks = torch.tensor(masks).permute(2, 0, 1)  # [n, h, w]
            else:
                RuntimeError("Type of `masks[0]` could not be interpreted: %s" % type(masks))
        elif isinstance(masks, BinaryMaskList):
            # just hard copy the BinaryMaskList instance's underlying data
            masks = masks.masks.clone()
        else:
            RuntimeError("Type of `masks` argument could not be interpreted:%s" % type(masks))

        if len(masks.shape) == 2:
            # if only a single instance mask is passed
            masks = masks[None]

        assert len(masks.shape) == 3
        assert masks.shape[1] == size[1], "%s != %s" % (masks.shape[1], size[1])
        assert masks.shape[2] == size[0], "%s != %s" % (masks.shape[2], size[0])

        self.masks = masks
        self.size = tuple(size)

    def transpose(self, method):
        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_masks = self.masks.flip(dim)
        return BinaryMaskList(flipped_masks, self.size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - TO_REMOVE)
        ymin = min(max(ymin, 0), current_height - TO_REMOVE)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + TO_REMOVE)
        ymax = max(ymax, ymin + TO_REMOVE)

        width, height = xmax - xmin, ymax - ymin
        cropped_masks = self.masks[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return BinaryMaskList(cropped_masks, cropped_size)

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
        resized_masks = torch.nn.functional.interpolate(
            input=self.masks[None].float(),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0].type_as(self.masks)
        resized_size = width, height
        return BinaryMaskList(resized_masks, resized_size)

    def crop_and_resize(self, box, size):
        device = self.masks.device
        if len(box.shape) == 1:
            box = box.unsqueeze(0)
        assert len(box.shape) == 2
        batch_inds = torch.arange(len(box), device=device).to(dtype=box.dtype)[:, None]
        rois = torch.cat([batch_inds, box], dim=1)  # Nx5

        bit_masks = self.masks.to(dtype=torch.float32)
        rois = rois.to(device=device)
        output = (
            roi_align(
                bit_masks[:, None, :, :], rois, size, 1.0, 0, True
            ).squeeze(1)
        )
        crop_resized_masks = output >= 0.5
        crop_resized_masks = crop_resized_masks.to(torch.float32)
        return BinaryMaskList(crop_resized_masks, size)

    # def set_size(self, size):
    #     return BinaryMaskList(self.masks, size)

    def convert_to_polygon(self):
        contours = self._findContours()
        return PolygonList(contours, self.size)

    def to(self, device):
        self.masks = self.masks.to(device)
        return self

    def _findContours(self):
        contours = []
        masks = self.masks.detach().numpy()
        for mask in masks:
            mask = cv2.UMat(mask)
            contour, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

            reshaped_contour = []
            for entity in contour:
                assert len(entity.shape) == 3
                assert entity.shape[1] == 1, "Hierarchical contours are not allowed"
                reshaped_contour.append(entity.reshape(-1).tolist())
            contours.append(reshaped_contour)
        return contours

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, index):
        # Probably it can cause some overhead
        # but preserves consistency
        masks = self.masks[index].clone()
        return BinaryMaskList(masks, self.size)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.masks))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


class PolygonList(object):
    """
    This class handles PolygonInstances for all objects in the image
    """

    def __init__(self, polygons, size):
        """
        Arguments:
            polygons:
                a list of list of lists of numbers. The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                object, and the third level to the polygon coordinates.

                OR

                a list of PolygonInstances.

                OR

                a PolygonList

            size: absolute image size

        """
        def _make_array(t):
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.asarray(t).astype("float64")

        def process_polygons(polygons_per_instance):
            assert isinstance(polygons_per_instance, list), type(polygons_per_instance)
            polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
            for polygon in polygons_per_instance:
                assert len(polygon) % 2 == 0 and len(polygon) >= 6
            return polygons_per_instance

        if isinstance(polygons, PolygonList):
            self.polygons = polygons.polygons
        else:
            assert isinstance(polygons, list), type(polygons)
            self.polygons = [process_polygons(polygons_per_instance)
                             for polygons_per_instance in polygons]
        self.size = tuple(size)

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT, FLIP_TOP_BOTTOM):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT and FLIP_TOP_BOTTOM implemented")

        all_flipped_polygons = []
        polygons = copy.deepcopy(self.polygons)
        for single_polygon in polygons:
            flipped_polygons = []
            width, height = self.size
            if method == FLIP_LEFT_RIGHT:
                dim = width
                idx = 0
            elif method == FLIP_TOP_BOTTOM:
                dim = height
                idx = 1

            for p in single_polygon:
                p[idx::2] = dim - p[idx::2] - TO_REMOVE
                flipped_polygons.append(p)
            all_flipped_polygons.append(flipped_polygons)

        return PolygonList(all_flipped_polygons, size=self.size)

    def move(self, gap):
        w, h = self.size[0] - gap[0], self.size[0] - gap[1]
        all_moved_polygons = []
        polygons = copy.deepcopy(self.polygons)
        for single_polygon in polygons:
            assert isinstance(gap, (list, tuple, torch.Tensor)), str(type(gap))

            # gap is assumed to be xy.
            # current_width, current_height = self.size
            gap_x, gap_y = map(float, gap)

            moved_polygons = []
            for p in single_polygon:
                p[0::2] = p[0::2] - gap_x
                p[1::2] = p[1::2] - gap_y
                moved_polygons.append(p)
            all_moved_polygons.append(moved_polygons)

        moved_size = w, h
        return PolygonList(all_moved_polygons, moved_size)

    def crop(self, box):
        w, h = box[2] - box[0], box[3] - box[1]
        all_cropped_polygons = []
        polygons = copy.deepcopy(self.polygons)
        for single_polygon in polygons:
            assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))

            # box is assumed to be xyxy
            current_width, current_height = self.size
            xmin, ymin, xmax, ymax = map(float, box)

            # assert xmin <= xmax and ymin <= ymax, str(box)
            xmin = min(max(xmin, 0), current_width - TO_REMOVE)
            ymin = min(max(ymin, 0), current_height - TO_REMOVE)

            xmax = min(max(xmax, 0), current_width)
            ymax = min(max(ymax, 0), current_height)

            xmax = max(xmax, xmin + TO_REMOVE)
            ymax = max(ymax, ymin + TO_REMOVE)

            # w, h = xmax - xmin, ymax - ymin

            cropped_polygons = []
            for p in single_polygon:
                p[0::2] = p[0::2] - xmin  # .clamp(min=0, max=w)
                p[1::2] = p[1::2] - ymin  # .clamp(min=0, max=h)
                cropped_polygons.append(p)
            all_cropped_polygons.append(cropped_polygons)
        cropped_size = w, h
        return PolygonList(all_cropped_polygons, cropped_size)

    def set_size(self, size):
        self.size = size
        return self

    def resize(self, size):
        all_resized_polygons = []
        polygons = copy.deepcopy(self.polygons)
        for single_polygon in polygons:
            try:
                iter(size)
            except TypeError:
                assert isinstance(size, (int, float))
                size = size, size

            ratios = tuple(float(s) / max(float(s_orig), 0.0001) for s, s_orig in zip(size, self.size))

            if ratios[0] == ratios[1]:
                ratio = ratios[0]
                scaled_polygons = [p * ratio for p in single_polygon]
            else:
                ratio_w, ratio_h = ratios
                scaled_polygons = []
                for p in single_polygon:
                    p[0::2] *= ratio_w
                    p[1::2] *= ratio_h
                    scaled_polygons.append(p)
            all_resized_polygons.append(scaled_polygons)

        resized_size = size
        return PolygonList(all_resized_polygons, resized_size)

    def crop_and_resize(self, boxes, size):
        boxes = boxes.to(torch.device("cpu"))
        cropped_resized_polygons = []
        assert len(self.polygons) == boxes.shape[0]
        polygons = copy.deepcopy(self.polygons)
        for single_polygon, box in zip(polygons, boxes):
            box = box.numpy()
            w, h = box[2] - box[0], box[3] - box[1]

            single_polygon = copy.deepcopy(single_polygon)
            for p in single_polygon:
                p[0::2] = p[0::2] - box[0]
                p[1::2] = p[1::2] - box[1]

            ratio_h = size[1] / max(h, 0.1)
            ratio_w = size[0] / max(w, 0.1)

            if ratio_h == ratio_w:
                for p in single_polygon:
                    p *= ratio_h
            else:
                for p in single_polygon:
                    p[0::2] *= ratio_w
                    p[1::2] *= ratio_h
            cropped_resized_polygons.append(single_polygon)
        return PolygonList(cropped_resized_polygons, size)

    def to(self, *args, **kwargs):
        return self

    def convert_to_binarymask(self):
        if len(self) > 0:
            def _convert_to_binarymask(polygons):
                width, height = self.size
                # formatting for COCO PythonAPI
                # polygons = [p.numpy() for p in polygons]
                rles = mask_utils.frPyObjects(polygons, height, width)
                rle = mask_utils.merge(rles)
                mask = mask_utils.decode(rle)
                mask = torch.from_numpy(mask)
                return mask
            masks = torch.stack([_convert_to_binarymask(p) for p in self.polygons])
        else:
            size = self.size
            masks = torch.empty([0, size[1], size[0]], dtype=torch.uint8)

        return BinaryMaskList(masks, size=self.size)

    def __len__(self):
        return len(self.polygons)

    def __getitem__(self, item):
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        else:
            # advanced indexing on a single dimension
            selected_polygons = []
            if isinstance(item, torch.Tensor) and (item.dtype == torch.uint8 or item.dtype == torch.bool):
                item = item.nonzero(as_tuple=False)
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                selected_polygons.append(self.polygons[i])
        return PolygonList(selected_polygons, size=self.size)

    def __iter__(self):
        return iter(self.polygons)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.polygons))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={})".format(self.size[1])
        return s


class Mask(object):
    """
    This class stores the mask for all objects in the image.
    It wraps BinaryMaskList and PolygonList conveniently.
    """

    def __init__(self, instances, size, mode="poly"):
        """
        Arguments:
            instances: two types
                (1) polygon
                (2) binary mask
            size: (width, height)
            mode: 'poly', 'mask'. if mode is 'mask', convert mask of any format to binary mask
        """

        assert isinstance(size, (list, tuple))
        assert len(size) == 2
        if isinstance(size[0], torch.Tensor):
            assert isinstance(size[1], torch.Tensor)
            size = size[0].item(), size[1].item()

        assert isinstance(size[0], (int, float))
        assert isinstance(size[1], (int, float))

        if mode == "poly":
            self.instances = PolygonList(instances, size)
        elif mode == "mask":
            self.instances = BinaryMaskList(instances, size)
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        self.mode = mode
        self.size = tuple(size)

    def transpose(self, method):
        flipped_instances = self.instances.transpose(method)
        return Mask(flipped_instances, self.size, self.mode)

    def move(self, gap):
        moved_instances = self.instances.move(gap)
        moved_size = (moved_instances.size[0] - gap[0], moved_instances.size[1] - gap[1])
        return Mask(moved_instances, moved_size, self.mode)

    def crop(self, box):
        cropped_instances = self.instances.crop(box)
        cropped_size = cropped_instances.size
        return Mask(cropped_instances, cropped_size, self.mode)

    def set_size(self, set_size):
        setted_instances = self.instances.set_size(set_size)
        return Mask(setted_instances, set_size, self.mode)

    def resize(self, size, *args, **kwargs):
        resized_instances = self.instances.resize(size)
        resized_size = size
        return Mask(resized_instances, resized_size, self.mode)

    def crop_and_resize(self, box, set_size):
        '''
        instances: poly (cpu) or mask (gpu)
        :param box: tensor (gpu)
        :param set_size: (w, h)
        '''
        cropped_instances = self.instances.crop_and_resize(box, set_size)
        cropped_size = cropped_instances.size
        return Mask(cropped_instances, cropped_size, self.mode)

    def to(self, device):
        self.instances.to(device)
        return self

    def convert(self, mode):
        if mode == self.mode:
            return self

        if mode == "poly":
            converted_instances = self.instances.convert_to_polygon()
        elif mode == "mask":
            converted_instances = self.instances.convert_to_binarymask()
        else:
            raise NotImplementedError("Unknown mode: %s" % str(mode))

        return Mask(converted_instances, self.size, mode)

    def get_mask_tensor(self):
        instances = self.instances
        if self.mode == "poly":
            instances = instances.convert_to_binarymask()
        # If there is only 1 instance
        return instances.masks.squeeze(0)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, item):
        selected_instances = self.instances.__getitem__(item)
        return Mask(selected_instances, self.size, self.mode)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_mask = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_mask
        raise StopIteration()

    next = __next__  # Python 2 compatibility

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self.instances))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        s += "mode={})".format(self.mode)
        return s
