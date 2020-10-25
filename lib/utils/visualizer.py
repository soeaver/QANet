import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_util
from collections import defaultdict

import lib.utils.colormap as colormap_utils
from lib.data.structures.keypoint import PersonKeypoints
from lib.utils.timer import Timer


class Visualizer(object):
    def __init__(self, cfg, im, dataset=None):

        self.cfg = cfg
        self.im = np.ascontiguousarray(np.asarray(im)[:, :, ::-1])  # BGR255
        self.dataset = dataset

        self._GRAY = [218, 227, 218]
        self._GREEN = [18, 127, 15]
        self._WHITE = [255, 255, 255]

    def vis_semseg_preds(self, semsegs=None, panos=None, label=None):
        timers = defaultdict(Timer)
        timers['show_semseg_masks'].tic()
        if panos is not None:
            img_resize = cv2.resize(np.array(self.im), (panos.shape[1], panos.shape[0]))
            segment_colors = self.get_colormap(self.cfg.SHOW_PANOS.COLORMAP)
            results = label['segments_info']
            pred_color = self.vis_segment_mask(panos, segment_colors, results=results, show_pano=True)
            self.im = cv2.addWeighted(img_resize, 1 - self.cfg.SHOW_PANOS.PANO_ALPHA, pred_color, self.cfg.SHOW_PANOS.PANO_ALPHA, 0)
        elif semsegs is not None:
            img_resize = cv2.resize(np.array(self.im), (semsegs.shape[1], semsegs.shape[0]))
            segment_colors = self.get_colormap(self.cfg.SHOW_SEMSEG.COLORMAP)
            color_mode = self.cfg.SHOW_SEMSEG.COLOR_MODE
            label_shift = self.cfg.SHOW_SEMSEG.LABEL_SHIFT
            ignore = self.cfg.SHOW_SEMSEG.IGNORE_LABEL
            pred_color = self.vis_segment_mask(semsegs, segment_colors, label_shift=label_shift, mode=color_mode, ignore=ignore)
            if label is not None:
                seg_color = self.vis_segment_mask(label, segment_colors, label_shift=label_shift, mode=color_mode, ignore=ignore)
                self.im = np.concatenate((img_resize, seg_color, pred_color), axis=1).astype(np.uint8)
            else:
                self.im = np.concatenate((img_resize, pred_color), axis=1).astype(np.uint8)
        else:
            raise NotImplementedError
        timers['show_semseg_masks'].toc()

        return self.im

    def vis_preds(self, boxes=None, classes=None, masks=None, keypoints=None, parsings=None, uvs=None, hiers=None,
                  semsegs=None, panos=None, panos_label=None):
        """Constructs a numpy array with the detections visualized."""
        timers = defaultdict(Timer)
        timers['bbox_prproc'].tic()

        if (semsegs is not None and self.cfg.SHOW_SEMSEG.ENABLED) or \
                (panos is not None and self.cfg.SHOW_PANOS.ENABLED):
            return self.vis_semseg_preds(semsegs, panos, panos_label)

        if hiers is not None and self.cfg.SHOW_HIER.ENABLED:
            classes = np.array(classes)
            boxes = boxes[classes == 0]
            classes = classes[classes == 0]

        if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < self.cfg.VIS_TH:
            return self.im

        if masks is not None and len(masks) > 0:
            bit_masks = mask_util.decode(masks)
        else:
            bit_masks = masks

        # get color map
        ins_colormap = self.get_colormap(self.cfg.SHOW_BOX.COLORMAP)

        # Display in largest to smallest order to reduce occlusion
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        sorted_inds = np.argsort(-areas)
        timers['bbox_prproc'].toc()

        instance_id = 1
        for i in sorted_inds:
            bbox = boxes[i, :-1]
            score = boxes[i, -1]
            if score < self.cfg.VIS_TH:
                continue

            # get instance color (box, class_bg)
            if self.cfg.SHOW_BOX.COLOR_SCHEME == 'category':
                ins_color = ins_colormap[classes[i]]
            elif self.cfg.SHOW_BOX.COLOR_SCHEME == 'instance':
                instance_id = instance_id % len(ins_colormap.keys())
                ins_color = ins_colormap[instance_id]
            else:
                ins_color = self._GREEN
            instance_id += 1

            # show box (on by default)
            if self.cfg.SHOW_BOX.ENABLED:
                if len(bbox) == 4:
                    timers['show_box'].tic()
                    self.vis_bbox((bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]), ins_color)
                    timers['show_box'].toc()
                elif len(bbox) == 5 and self.cfg.SHOW_BOX.SHOW_ROTATED_BOX:
                    timers['show_box'].tic()
                    self.vis_rotated_bbox(bbox, ins_color)
                    timers['show_box'].toc()
                else:
                    raise RuntimeError("Check the box format")

            # show class (on by default)
            if self.cfg.SHOW_CLASS.ENABLED:
                timers['show_class'].tic()
                class_str = self.get_class_string(classes[i], score)
                self.vis_class((bbox[0], bbox[1] - 2), class_str, ins_color)
                timers['show_class'].toc()

            show_masks = True if masks is not None and self.cfg.SHOW_MASK.ENABLED and len(masks) > i else False
            show_kpts = True if keypoints is not None and self.cfg.SHOW_KPS.ENABLED and len(keypoints) > i else False
            show_parss = True if parsings is not None and self.cfg.SHOW_PARSS.ENABLED and len(parsings) > i else False
            show_uvs = True if uvs is not None and self.cfg.SHOW_UV.ENABLED and len(uvs) > i else False
            show_hiers = True if hiers is not None and self.cfg.SHOW_HIER.ENABLED and len(hiers) > i else False

            # show mask
            if show_masks:
                timers['show_masks'].tic()
                self.vis_mask(bit_masks[..., i], ins_color, show_parss=show_parss)
                timers['show_masks'].toc()

            # show keypoints
            if show_kpts:
                timers['show_kpts'].tic()
                self.vis_keypoints(keypoints[i], show_parss=show_parss)
                timers['show_kpts'].toc()

            # show parsings
            if show_parss:
                timers['show_parss'].tic()
                parss_colormap = self.get_colormap(self.cfg.SHOW_PARSS.COLORMAP)
                self.vis_parsing(parsings[i], parss_colormap, show_masks=show_masks)
                timers['show_parss'].toc()

            # show uvs
            if show_uvs:
                timers['show_uvs'].tic()
                self.vis_uv(uvs[i], bbox)
                timers['show_uvs'].toc()

            # show hiers
            if show_hiers:
                timers['show_hiers'].tic()
                self.vis_hier(hiers[i], ins_color)
                timers['show_hiers'].toc()

        # for k, v in timers.items():
        #     print(' | {}: {:.3f}s'.format(k, v.total_time))

        return self.im

    def get_class_string(self, class_index, score=None):
        class_text = self.dataset.classes[class_index] if self.dataset is not None else 'id{:d}'.format(class_index)
        if score is None:
            return class_text
        return class_text + ' {:0.2f}'.format(score).lstrip('0')

    def get_colormap(self, colormap_type, rgb=False):
        colormap = eval('colormap_utils.{}'.format(colormap_type))
        if rgb:
            colormap = colormap_utils.dict_bgr2rgb(colormap)
        return colormap

    def vis_bbox(self, bbox, bbox_color):
        """Visualizes a bounding box."""
        (x0, y0, w, h) = bbox
        x1, y1 = int(x0 + w), int(y0 + h)
        x0, y0 = int(x0), int(y0)
        cv2.rectangle(self.im, (x0, y0), (x1, y1), bbox_color, thickness=self.cfg.SHOW_BOX.BORDER_THICK)

    def vis_rotated_bbox(self, rotated_bbox, bbox_color):
        """Visualizes a rotated bounding box."""
        (x0, y0, w, h, a) = rotated_bbox
        h_bbox = [(x0 - w / 2, y0 - h / 2),
                  (x0 + w / 2, y0 - h / 2),
                  (x0 + w / 2, y0 + h / 2),
                  (x0 - w / 2, y0 + h / 2)]
        import math
        a = a * math.pi / 180
        cos = math.cos(a)
        sin = math.sin(a)
        o_bbox = []
        for i in range(len(h_bbox)):
            x_i = int(sin * (h_bbox[i][1] - y0) + cos * (h_bbox[i][0] - x0) + x0)
            y_i = int(cos * (h_bbox[i][1] - y0) - sin * (h_bbox[i][0] - x0) + y0)
            o_bbox.append((x_i, y_i))
        for j in range(len(o_bbox)):
            cv2.line(self.im, o_bbox[j], o_bbox[(j + 1) % len(o_bbox)],
                     bbox_color, thickness=self.cfg.SHOW_BOX.BORDER_THICK)

    def vis_class(self, pos, class_str, bg_color):
        """Visualizes the class."""
        font_color = self.cfg.SHOW_CLASS.COLOR
        font_scale = self.cfg.SHOW_CLASS.FONT_SCALE

        x0, y0 = int(pos[0]), int(pos[1])
        # Compute text size.
        txt = class_str
        font = cv2.FONT_HERSHEY_SIMPLEX
        ((txt_w, txt_h), _) = cv2.getTextSize(txt, font, font_scale, 1)
        # Place text background.
        back_tl = x0, y0 - int(1.3 * txt_h)
        back_br = x0 + txt_w, y0
        cv2.rectangle(self.im, back_tl, back_br, bg_color, -1)
        # Show text.
        txt_tl = x0, y0 - int(0.3 * txt_h)
        cv2.putText(self.im, txt, txt_tl, font, font_scale, font_color, lineType=cv2.LINE_AA)

    def vis_mask(self, mask, bbox_color, show_parss=False):
        """Visualizes a single binary mask."""
        self.im = self.im.astype(np.float32)
        idx = np.nonzero(mask)

        border_color = self.cfg.SHOW_MASK.BORDER_COLOR
        border_thick = self.cfg.SHOW_MASK.BORDER_THICK

        mask_color = bbox_color if self.cfg.SHOW_MASK.MASK_COLOR_FOLLOW_BOX else _WHITE
        mask_color = np.asarray(mask_color)
        mask_alpha = self.cfg.SHOW_MASK.MASK_ALPHA

        try:
            _, contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        except:
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if self.cfg.SHOW_MASK.SHOW_BORDER:
            cv2.drawContours(self.im, contours, -1, border_color, border_thick, cv2.LINE_AA)

        if not show_parss:
            self.im[idx[0], idx[1], :] *= 1.0 - mask_alpha
            self.im[idx[0], idx[1], :] += mask_alpha * mask_color

        self.im.astype(np.uint8)

    def vis_keypoints(self, kps, show_parss=False):
        """Visualizes keypoints (adapted from vis_one_image).
        kps has shape (4, #keypoints) where 4 rows are (x, y, logit, prob).
        """
        dataset_keypoints = PersonKeypoints.NAMES
        kp_lines = PersonKeypoints.CONNECTIONS

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]
        if show_parss:
            colors = [self.cfg.SHOW_KPS.KPS_COLOR_WITH_PARSING for c in colors]
        else:
            colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

        # Perform the drawing on a copy of the image, to allow for blending.
        # kp_mask = np.copy(self.im)

        # Draw mid shoulder / mid hip first for better visualization.
        mid_shoulder = (kps[:2, dataset_keypoints.index('right_shoulder')] +
                        kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
        sc_mid_shoulder = np.minimum(
            kps[2, dataset_keypoints.index('right_shoulder')],
            kps[2, dataset_keypoints.index('left_shoulder')])
        mid_hip = (kps[:2, dataset_keypoints.index('right_hip')] +
                   kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
        sc_mid_hip = np.minimum(
            kps[2, dataset_keypoints.index('right_hip')],
            kps[2, dataset_keypoints.index('left_hip')])
        nose_idx = dataset_keypoints.index('nose')
        if sc_mid_shoulder > self.cfg.SHOW_KPS.KPS_TH and kps[2, nose_idx] > self.cfg.SHOW_KPS.KPS_TH:
            cv2.line(self.im, tuple(mid_shoulder), tuple(kps[:2, nose_idx]), color=colors[len(kp_lines)],
                     thickness=self.cfg.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)
        if sc_mid_shoulder > self.cfg.SHOW_KPS.KPS_TH and sc_mid_hip > self.cfg.SHOW_KPS.KPS_TH:
            cv2.line(self.im, tuple(mid_shoulder), tuple(mid_hip), color=colors[len(kp_lines) + 1],
                     thickness=self.cfg.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)

        # Draw the keypoints.
        for l in range(len(kp_lines)):
            i1 = kp_lines[l][0]
            i2 = kp_lines[l][1]
            p1 = kps[0, i1], kps[1, i1]
            p2 = kps[0, i2], kps[1, i2]
            if kps[2, i1] > self.cfg.SHOW_KPS.KPS_TH and kps[2, i2] > self.cfg.SHOW_KPS.KPS_TH:
                cv2.line(self.im, p1, p2, color=colors[l],
                         thickness=self.cfg.SHOW_KPS.LINK_THICK, lineType=cv2.LINE_AA)
            if kps[2, i1] > self.cfg.SHOW_KPS.KPS_TH:
                cv2.circle(self.im, p1, radius=self.cfg.SHOW_KPS.CIRCLE_RADIUS, color=colors[l],
                           thickness=self.cfg.SHOW_KPS.CIRCLE_THICK, lineType=cv2.LINE_AA)
            if kps[2, i2] > self.cfg.SHOW_KPS.KPS_TH:
                cv2.circle(self.im, p2, radius=self.cfg.SHOW_KPS.CIRCLE_RADIUS, color=colors[l],
                           thickness=self.cfg.SHOW_KPS.CIRCLE_THICK, lineType=cv2.LINE_AA)

        # Blend the keypoints.
        cv2.addWeighted(self.im, 1.0 - self.cfg.SHOW_KPS.KPS_ALPHA, self.im, self.cfg.SHOW_KPS.KPS_ALPHA, 0)

    def vis_parsing(self, parsing, colormap, show_masks=True):
        """Visualizes a single binary parsing."""
        self.im = self.im.astype(np.float32)
        idx = np.nonzero(parsing)

        parsing_alpha = self.cfg.SHOW_PARSS.PARSING_ALPHA
        colormap = colormap_utils.dict2array(colormap)
        parsing_color = colormap[parsing.astype(np.int)]

        border_color = self.cfg.SHOW_PARSS.BORDER_COLOR
        border_thick = self.cfg.SHOW_PARSS.BORDER_THICK

        self.im[idx[0], idx[1], :] *= 1.0 - parsing_alpha
        # self.im[idx[0], idx[1], :] += alpha * parsing_color
        self.im += parsing_alpha * parsing_color

        if self.cfg.SHOW_PARSS.SHOW_BORDER and not show_masks:
            try:
                _, contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            except:
                contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(self.im, contours, -1, border_color, border_thick, cv2.LINE_AA)

        self.im.astype(np.uint8)

    def vis_uv(self, uv, bbox):
        border_thick = self.cfg.SHOW_UV.BORDER_THICK
        grid_thick = self.cfg.SHOW_UV.GRID_THICK
        lines_num = self.cfg.SHOW_UV.LINES_NUM

        uv = np.transpose(uv, (1, 2, 0))
        uv = cv2.resize(uv, (int(bbox[2] - bbox[0] + 1), int(bbox[3] - bbox[1] + 1)), interpolation=cv2.INTER_LINEAR)
        roi_img = self.im[int(bbox[1]):int(bbox[3] + 1), int(bbox[0]):int(bbox[2] + 1), :]

        roi_img_resize = cv2.resize(roi_img, (2 * roi_img.shape[1], 2 * roi_img.shape[0]),
                                    interpolation=cv2.INTER_LINEAR)

        I = uv[:, :, 0]
        for i in range(1, 25):
            if len(I[I == i]) == 0:
                continue

            u = np.zeros_like(I)
            v = np.zeros_like(I)
            u[I == i] = uv[:, :, 1][I == i]
            v[I == i] = uv[:, :, 2][I == i]

            for ind in range(1, lines_num):
                thred = 1.0 * ind / lines_num
                _, thresh = cv2.threshold(u, u.min() + thred * (u.max() - u.min()), 255, 0)
                dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
                dist_transform = np.uint8(dist_transform)

                try:
                    _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                except:
                    contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                contours = [(col * 2) for col in contours]
                cv2.drawContours(roi_img_resize, contours, -1, ((1 - thred) * 255, thred * 255, thred * 200),
                                 grid_thick)

                _, thresh = cv2.threshold(v, v.min() + thred * (v.max() - v.min()), 255, 0)
                dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
                dist_transform = np.uint8(dist_transform)

                try:
                    _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                except:
                    contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                contours = [(col * 2) for col in contours]
                cv2.drawContours(roi_img_resize, contours, -1, (thred * 255, (1 - thred) * 255, thred * 200),
                                 grid_thick)

        _, thresh = cv2.threshold(I, 0.5, 255, 0)
        dist_transform = cv2.distanceTransform(np.uint8(thresh), cv2.DIST_L2, 0)
        dist_transform = np.uint8(dist_transform)
        try:
            _, contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        except:
            contours, _ = cv2.findContours(dist_transform, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        contours = [(col * 2) for col in contours]
        cv2.drawContours(roi_img_resize, contours, -1, (70, 150, 0), border_thick)

        roi_img[:] = cv2.resize(roi_img_resize, (roi_img.shape[1], roi_img.shape[0]), interpolation=cv2.INTER_LINEAR)[:]

        return self.im

    def vis_hier(self, hier, bbox_color):
        border_thick = self.cfg.SHOW_HIER.BORDER_THICK
        N = len(hier) // 5
        for i in range(N):
            if hier[i * 5 + 4] > 0:
                cv2.rectangle(
                    self.im,
                    (int(hier[i * 5]), int(hier[i * 5 + 1])),
                    (int(hier[i * 5 + 2]), int(hier[i * 5 + 3])),
                    bbox_color,
                    thickness=border_thick
                )

    def vis_segment_mask(self, labelmap, colors, label_shift=0, mode='BGR', ignore=0, results=None, show_pano=False):
        """Visualize segmentation."""
        labelmap_bgr = np.zeros((labelmap.shape[0], labelmap.shape[1], 3), dtype=np.uint8)
        if show_pano and results is not None:
            id_pair = self.dataset.json_pano_id_to_contiguous_category_id
            boxes = []
            classes = []
            for seg_info in results:
                if not isinstance(labelmap, np.ndarray):
                    if labelmap.is_cuda:
                        labelmap = labelmap.cpu()
                    labelmap = labelmap.numpy()
                color = colors[seg_info['category_id']]
                label = seg_info['id']
                if seg_info['isthing']:
                    color += np.random.randint(low=-70, high=70, size=3)
                    color = list(np.maximum(0, np.minimum(255, color)))
                    binary_map = (labelmap == label)[:,:,np.newaxis].astype(np.uint8)
                    x, y, w, h = cv2.boundingRect(binary_map)
                    boxes.append((x, y, w, h))
                    classes.append(seg_info['category_id'])
                labelmap_bgr += (labelmap == label)[:, :, np.newaxis] * np.tile(
                    np.asarray(color, dtype=np.uint8), (labelmap.shape[0], labelmap.shape[1], 1))
            self.im = labelmap_bgr
            for box, cls in zip(boxes, classes):
                self.vis_bbox(box, self._GREEN)
                class_str = self.get_class_string(id_pair[cls])
                self.vis_class((box[0], box[1] - 2), class_str, self._GREEN)
        else:
            if not isinstance(labelmap, np.ndarray):
                if labelmap.is_cuda:
                    labelmap = labelmap.cpu()
            for label in np.unique(labelmap):
                if label < 0:
                    labelmap_bgr += (labelmap == label)[:, :, np.newaxis] * np.tile(
                        np.asarray([ignore, ignore, ignore], dtype=np.uint8), (labelmap.shape[0], labelmap.shape[1], 1))
                    continue
                labelmap_bgr += (labelmap == label)[:, :, np.newaxis] * np.tile(
                    np.asarray(colors[np.uint8(label - label_shift)], dtype=np.uint8), (labelmap.shape[0], labelmap.shape[1], 1))

        if mode == 'RGB':
            return labelmap_bgr[:, :, ::-1]
        else:
            return labelmap_bgr
