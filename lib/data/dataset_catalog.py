import os.path as osp

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'
_FIELDS = 'extra_fields'

# Available datasets
COMMON_DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg']}
    },
    'cityscapes_fine_instanceonly_seg_val': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        # use filtered validation as there is an issue converting contours
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg']}
    },
    'cityscapes_fine_instanceonly_seg_test': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg']}
    },
    'coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'coco_2017_train_lvisanno': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_train_cocofied.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'coco_2017_val_lvisanno': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_val_cocofied.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'coco_stuff_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json',
        _FIELDS:
            {'ann_types': ['semseg']}
    },
    'coco_stuff_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json',
        _FIELDS:
            {'ann_types': ['semseg']}
    },
    'keypoints_coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'keypoints']}
    },
    'keypoints_coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'keypoints']}
    },
    'keypoints_coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'keypoints']}
    },
    'keypoints_coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'keypoints']}
    },
    'dense_coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_train2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'uv'],
             'uv_root': _DATA_DIR + '/coco/annotations/'}
    },
    'dense_coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_val2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'uv'],
             'uv_root': _DATA_DIR + '/coco/annotations/'}
    },
    'dense_coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'uv'],
             'uv_root': _DATA_DIR + '/coco/annotations/'}
    },
    'lvis_v0.5_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask'],
             'seg_root': _DATA_DIR + '/coco/images/stuffthingmaps/train2017'}
    },
    'lvis_v0.5_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_val_2017.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask'],
             'seg_root': _DATA_DIR + '/coco/images/stuffthingmaps/val2017'}
    },
    'lvis_v0.5_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_image_info_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'lvis_v1_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'lvis_v1_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'lvis_v1_train-val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_train_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'lvis_v1_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_image_info_test_dev.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'lvis_v1_test-challenge': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_image_info_test_challenge.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask']}
    },
    'CIHP_train': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_train.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/CIHP/Training/Category_ids',
             'label_shift': 0},
    },
    'CIHP_val': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/CIHP/Validation/Category_ids',
             'label_shift': 0},
    },
    'CIHP_test': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_test.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             # 'seg_root': _DATA_DIR + '/CIHP/Testing/Category_ids',  # no gt seg
             'label_shift': 0},
    },
    'LIP_train': {
        _IM_DIR:
            _DATA_DIR + '/LIP/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/LIP/Training/Category_ids',
             'label_shift': 0}
    },
    'LIP_val': {
        _IM_DIR:
            _DATA_DIR + '/LIP/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/LIP/Validation/Category_ids',
             'label_shift': 0}
    },
    'LIP_test': {
        _IM_DIR:
            _DATA_DIR + '/LIP/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'label_shift': 0}
    },
    'ATR_train': {
        _IM_DIR:
            _DATA_DIR + '/ATR/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([9, 10], [12, 13], [14, 15]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/ATR/Training/Category_ids',
             'label_shift': 0}
    },
    'ATR_val': {
        _IM_DIR:
            _DATA_DIR + '/ATR/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([9, 10], [12, 13], [14, 15]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/ATR/Validation/Category_ids',
             'label_shift': 0}
    },
    'VIP_Fine_train': {
        _IM_DIR:
            _DATA_DIR + '/VIP/VIP_Fine/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/VIP/VIP_Fine/annotations/VIP_Fine_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/VIP/VIP_Fine/Training/Category_ids',
             'label_shift': 0}
    },
    'VIP_Fine_val': {
        _IM_DIR:
            _DATA_DIR + '/VIP/VIP_Fine/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/VIP/VIP_Fine/annotations/VIP_Fine_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/VIP/VIP_Fine/Validation/Category_ids',
             'label_shift': 0}
    },
    'VIP_Fine_test': {
        _IM_DIR:
            _DATA_DIR + '/VIP/VIP_Fine/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/VIP/VIP_Fine/annotations/VIP_Fine_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             # 'seg_root': _DATA_DIR + '/VIP/VIP_Fine/Testing/Category_ids',  # no gt seg
             'label_shift': 0}
    },
    'MHP-v2_train': {
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'parsing'],
             'seg_root': _DATA_DIR + '/MHP-v2/Training/Category_ids',
             'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33]),
             'label_shift': 0},
    },
    'MHP-v2_val': {
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'parsing'],
             'seg_root': _DATA_DIR + '/MHP-v2/Validation/Category_ids',
             'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33]),
             'label_shift': 0},
    },
    'MHP-v2_test': {
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'parsing'],
             # 'seg_root': _DATA_DIR + '/MHP-v2/Testing/Category_ids',  # no gt seg
             'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33]),
             'label_shift': 0},
    },
    'PASCAL-Person-Part_train': {
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/PASCAL-Person-Part_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/PASCAL-Person-Part/Training/Category_ids',
             'label_shift': 0}
    },
    'PASCAL-Person-Part_test': {
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/PASCAL-Person-Part_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/PASCAL-Person-Part/Testing/Category_ids',
             'label_shift': 0}
    },
    'LaPa_compact-box_train': {
        _IM_DIR:
            _DATA_DIR + '/LaPa/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/LaPa/annotations/LaPa_compact-box_train.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([2, 3], [4, 5]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/LaPa/Training/Category_ids',
             'label_shift': 0}
    },
    'LaPa_compact-box_val': {
        _IM_DIR:
            _DATA_DIR + '/LaPa/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/LaPa/annotations/LaPa_compact-box_val.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([2, 3], [4, 5]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/LaPa/Validation/Category_ids',
             'label_shift': 0}
    },
    'LaPa_compact-box_test': {
        _IM_DIR:
            _DATA_DIR + '/LaPa/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/LaPa/annotations/LaPa_compact-box_test.json',
        _FIELDS:
            {'ann_types': ['bbox', 'mask', 'semseg', 'parsing'],
             'flip_map': ([2, 3], [4, 5]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/LaPa/Testing/Category_ids',
             'label_shift': 0}
    },
}
