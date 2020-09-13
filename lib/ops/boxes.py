from lib.ops import _C

BOX_VOTING_METHODS = {'ID': 0, 'TEMP_AVG': 1, 'AVG': 2, 'IOU_AVG': 3, 'GENERALIZED_AVG': 4, 'QUASI_SUM': 5}


def box_voting(top_boxes, top_scores, all_boxes, all_scores, overlap_thresh, method='ID', beta=1.0):
    """Apply bounding-box voting to refine top dets by voting with all dets.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """

    assert method in BOX_VOTING_METHODS, 'Unknown box_voting method: {}'.format(method)

    return _C.box_voting(
        top_boxes,
        top_scores,
        all_boxes,
        all_scores,
        BOX_VOTING_METHODS[method],
        beta,
        overlap_thresh
    )


def box_ml_voting(top_boxes, top_scores, top_labels, all_boxes, all_scores, all_labels, overlap_thresh, method='ID', beta=1.0):
    """Apply bounding-box voting to refine top dets by voting with all dets.
    See: https://arxiv.org/abs/1505.01749. Optional score averaging (not in the
    referenced  paper) can be applied by setting `scoring_method` appropriately.
    """

    assert method in BOX_VOTING_METHODS, 'Unknown box_voting method: {}'.format(method)

    return _C.box_ml_voting(
        top_boxes,
        top_scores,
        top_labels,
        all_boxes,
        all_scores,
        all_labels,
        BOX_VOTING_METHODS[method],
        beta,
        overlap_thresh
    )


def box_iou(boxes1, boxes2):
    return _C.box_iou(boxes1, boxes2)


def box_iou_rotated(boxes1, boxes2):
    return _C.box_iou_rotated(boxes1, boxes2)
