import json
import os
import cv2
import numpy as np
import multiprocessing as mp

from PIL import Image
from collections import defaultdict

from lib.utils.misc import logging_rank


class PanopticEvaluator:
    """
    Evaluate panoptic, implement following "evaluate.py" in panopticapi.
    """
    def __init__(self, pred_results, gt_dir, json_result_file, json_gt_file, pred_dir=None):
        self.stats = None
        self.categories = None
        self.pred_results = pred_results
        self.json_result_file = json_result_file
        self.json_gt_file = json_gt_file
        self.gt_dir = gt_dir
        self.pred_dir = json_result_file.replace('.json', '') if pred_dir is None else pred_dir
        self.metrics = [("All", None), ("Things", True), ("Stuff", False)]

    def evaluate(self):
        """
        Evaluate panoptic predictions.Only support coco evaluation for now.
        """
        logging_rank('Evaluating panoptic predictions...')
        os.makedirs(self.pred_dir, exist_ok=True)

        logging_rank('Saving predicted panoptic PNG to {}...'.format(self.pred_dir))
        for pred in self.pred_results:
            with open(os.path.join(self.pred_dir, pred["file_name"]), 'wb') as f:
                f.write(pred.pop('png_data'))

        with open(self.json_gt_file, "r") as f:
            json_data = json.load(f)
            self.categories = {el['id']: el for el in json_data['categories']}
            json_data['annotations'] = self.pred_results

        with open(self.json_result_file, 'w') as f:
            json.dump(json_data, f)

        logging_rank('Computing metrics ...')
        self.stats = self._pq_compute()

    def _pq_compute(self):
        pq_stat = PQStat()
        cpu_num = mp.cpu_count()
        workers = mp.Pool(processes=cpu_num)
        matched_annotations_list = []
        processes = []
        with open(self.json_gt_file, 'r') as f:
            gt_json = json.load(f)

        pred_annotations = {el['image_id']: el for el in self.pred_results}
        for gt_ann in gt_json['annotations']:
            image_id = gt_ann['image_id']
            if image_id not in pred_annotations:
                raise Exception('No prediction for the image with id: {}'.format(image_id))
            matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

        annotations_split = np.array_split(matched_annotations_list, cpu_num)
        for proc_id, annotation_set in enumerate(annotations_split):
            p = workers.apply_async(
                self._pq_compute_single_core, (proc_id, annotation_set)
            )
            processes.append(p)
        for p in processes:
            pq_stat += p.get()

        return pq_stat

    def _pq_compute_single_core(self, proc_id, annotation_set):
        pq_stat = PQStat()
        void = 0
        offset = 256 * 256 * 256

        for gt_ann, pred_ann in annotation_set:
            gt_segms = {el['id']: el for el in gt_ann['segments_info']}
            pred_segms = {el['id']: el for el in pred_ann['segments_info']}
            pan_gt = np.array(Image.open(os.path.join(self.gt_dir, gt_ann['file_name'])), dtype=np.uint32)
            pan_gt = rgb2id(pan_gt)
            pan_pred = np.array(Image.open(os.path.join(self.pred_dir, pred_ann['file_name'])), dtype=np.uint32)
            pan_pred = rgb2id(pan_pred)

            # predicted segments area calculation + prediction sanity checks
            pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
            labels, labels_cnt = np.unique(pan_pred, return_counts=True)
            for label, label_cnt in zip(labels, labels_cnt):
                if label not in pred_segms:
                    if label == void:
                        continue
                    raise KeyError(
                        'In the image with ID {} segment with ID {} is presented in PNG and not '
                        'presented in JSON.'.format(gt_ann['image_id'], label)
                    )
                pred_segms[label]['area'] = label_cnt
                pred_labels_set.remove(label)
                if pred_segms[label]['category_id'] not in self.categories:
                    raise KeyError('In the image with ID {} segment with ID {} has unknown category_id'
                                   ' {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id'])
                                   )
            if len(pred_labels_set) != 0:
                raise KeyError(
                    'In the image with ID {} the following segment IDs {} are presented in JSON '
                    'and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set))
                )

            # confusion matrix calculation
            pan_gt_pred = pan_gt.astype(np.uint64) * offset + pan_pred.astype(np.uint64)
            gt_pred_map = {}
            labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // offset
                pred_id = label % offset
                gt_pred_map[(gt_id, pred_id)] = intersection

            # count all matched pairs
            gt_matched = set()
            pred_matched = set()
            for label_tuple, intersection in gt_pred_map.items():
                gt_label, pred_label = label_tuple
                if gt_label not in gt_segms:
                    continue
                if pred_label not in pred_segms:
                    continue
                if gt_segms[gt_label]['iscrowd'] == 1:
                    continue
                if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                    continue
                union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get(
                    (void, pred_label), 0)
                iou = intersection / union
                if iou > 0.5:
                    pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                    pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                    gt_matched.add(gt_label)
                    pred_matched.add(pred_label)

            # count false positives
            crowd_labels_dict = {}
            for gt_label, gt_info in gt_segms.items():
                if gt_label in gt_matched:
                    continue
                # crowd segments are ignored
                if gt_info['iscrowd'] == 1:
                    crowd_labels_dict[gt_info['category_id']] = gt_label
                    continue
                pq_stat[gt_info['category_id']].fn += 1

            # count false positives
            for pred_label, pred_info in pred_segms.items():
                if pred_label in pred_matched:
                    continue
                # intersection of the segment with void
                intersection = gt_pred_map.get((void, pred_label), 0)
                # plus intersection with corresponding CROWD region if it exists
                if pred_info['category_id'] in crowd_labels_dict:
                    intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
                # predicted segment is ignored if more than half of the segment correspond to void and CROWD regions
                if intersection / pred_info['area'] > 0.5:
                    continue
                pq_stat[pred_info['category_id']].fp += 1

        return pq_stat

    def accumulate(self):
        logging_rank("Accumulating results ...")
        results = {}
        for name, isthing in self.metrics:
            results[name], per_class_results = self.stats.pq_average(self.categories, isthing=isthing)
            if name == 'All':
                results['per_class'] = per_class_results

        self.stats = results

    def summarize(self):
        logging_rank("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
        logging_rank("-" * (10 + 7 * 4))
        for name, _isthing in self.metrics:
            logging_rank("{:10s}| {:5.1f}  {:5.1f}  {:5.1f} {:5d}".format(
                name,
                100 * self.stats[name]['pq'],
                100 * self.stats[name]['sq'],
                100 * self.stats[name]['rq'],
                self.stats[name]['n'])
            )


class PQStatCat():
    def __init__(self):
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

        return {'pq': pq / n, 'sq': sq / n, 'rq': rq / n, 'n': n}, per_class_results


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color