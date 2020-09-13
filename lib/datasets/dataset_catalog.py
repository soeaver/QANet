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
    },
    'cityscapes_fine_instanceonly_seg_val': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        # use filtered validation as there is an issue converting contours
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_val.json',
    },
    'cityscapes_fine_instanceonly_seg_test': {
        _IM_DIR:
            _DATA_DIR + '/CityScape/images',
        _ANN_FN:
            _DATA_DIR + '/CityScape/annotations/instancesonly_filtered_gtFine_test.json',
    },
    'coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
    },
    'coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
    },
    'coco_2017_train_lvisanno': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_train_cocofied.json',
    },
    'coco_2017_val_lvisanno': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_val_cocofied.json',
    },
    'coco_stuff_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json'
    },
    'coco_stuff_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'keypoints_coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
    },
    'dense_coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_train2017.json',
    },
    'dense_coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_val2017.json',
    },
    'dense_coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_test.json',
    },
    'voc_2007_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/Json_Annos/voc_2007_train.json',
    },
    'voc_2007_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2007_trainval/Json_Annos/voc_2007_val.json',
    },
    'voc_2007_te-st': {  # new addition by wzh, 'test' will not be evaluated
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2007_test/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2007_test/Json_Annos/voc_2007_test.json',
    },
    'voc_2012_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/Json_Annos/voc_2012_train.json',
    },
    'voc_2012_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2012_trainval/Json_Annos/voc_2012_val.json',
    },
    'voc_2012_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/pascal_voc/VOC2012_test/JPEGImages',
        _ANN_FN:
            _DATA_DIR + '/pascal_voc/VOC2012_test/Json_Annos/voc_2012_test.json',
    },
    'CIHP_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP_v2/Training/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP_v2/annotations/CIHP_train.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'seg_root': _DATA_DIR + '/CIHP_v2/Training/Category_ids'},
    },
    'CIHP_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP_v2/Validation/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP_v2/annotations/CIHP_val.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'seg_root': _DATA_DIR + '/CIHP_v2/Validation/Category_ids'},
    },
    'CIHP_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP_v2/Testing/Images',
        _ANN_FN:
            _DATA_DIR + '/CIHP_v2/annotations/CIHP_test.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19])},
    },
    'VIP_Fine_train': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/ATEN/VIP_Fine/train_img',
        _ANN_FN:
            _DATA_DIR + '/ATEN/VIP_Fine/annotations/VIP_Fine_train.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19])},
    },
    'VIP_Fine_val': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/ATEN/VIP_Fine/val_img',
        _ANN_FN:
            _DATA_DIR + '/ATEN/VIP_Fine/annotations/VIP_Fine_val.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19])},
    },
    'VIP_Fine_test': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/ATEN/VIP_Fine/test_img',
        _ANN_FN:
            _DATA_DIR + '/ATEN/VIP_Fine/annotations/VIP_Fine_test.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19])},
    },
    'MHP-v2_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/train_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_train.json',
        _FIELDS:
            {'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])},
    },
    'MHP-v2_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/val_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_val.json',
        _FIELDS:
            {'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])},
    },
    'MHP-v2_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_all.json',
        _FIELDS:
            {'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])},
    },
    'MHP-v2_test_inter_top10': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_inter_top10.json',
        _FIELDS:
            {'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])},
    },
    'MHP-v2_test_inter_top20': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_inter_top20.json',
        _FIELDS:
            {'flip_map': ([5, 6], [7, 8], [22, 23], [24, 25], [26, 27], [28, 29], [30, 31], [32, 33])},
    },
    'PASCAL-Person-Part_train': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/train_img',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/pascal_person_part_train.json',
    },
    'PASCAL-Person-Part_test': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/test_img',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/pascal_person_part_test.json',
    },
    'WIDER_train': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/Wider_Face/WIDER_train',
        _ANN_FN:
            _DATA_DIR + '/Wider_Face/Json_Annos/WIDER_train.json',
    },
    'WIDER_val': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/Wider_Face/WIDER_val',
        _ANN_FN:
            _DATA_DIR + '/Wider_Face/Json_Annos/WIDER_val.json',
    },
    'FDDB': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/FDDB',
        _ANN_FN:
            _DATA_DIR + '/FDDB/Annotations/FDDB.json',
    },
    'lvis_v0.5_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_train.json',
        _FIELDS:
            {'seg_root': _DATA_DIR + '/coco/images/stuffthingmaps/train2017'}
    },
    'lvis_v0.5_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_val_2017.json',
        _FIELDS:
            {'seg_root': _DATA_DIR + '/coco/images/stuffthingmaps/val2017'}
    },
    'lvis_v0.5_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_image_info_test.json',
    },
    'lvis_v0.5_train_cocostuff': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_train.json',
        _FIELDS:
            {'seg_json': _DATA_DIR + '/coco/annotations/stuff_train2017.json',
             'label_shift': 0}
    },
    'lvis_v0.5_val_cocostuff': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_val_2017.json',
        _FIELDS:
            {'seg_json': _DATA_DIR + '/coco/annotations/stuff_val2017.json',
             'label_shift': 0}
    },
    'lvis_v1_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_train.json',
    },
    'lvis_v1_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_val.json',
    },
    'lvis_v1_train-val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_train_val.json',
    },
    'lvis_v1_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_image_info_test_dev.json',
    },
    'lvis_v1_test-challenge': {
        _IM_DIR:
            _DATA_DIR + '/coco/images',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v1_image_info_test_challenge.json',
    },
    'objects365_train': {
        _IM_DIR:
            _DATA_DIR + '/obj365/images/train',
        _ANN_FN:
            _DATA_DIR + '/obj365/annotations/objects365_train.json',
    },
    'objects365_val': {
        _IM_DIR:
            _DATA_DIR + '/obj365/images/val',
        _ANN_FN:
            _DATA_DIR + '/obj365/annotations/objects365_val.json',
    },
    'objects365_test': {
        _IM_DIR:
            _DATA_DIR + '/obj365/images/test',
        _ANN_FN:
            _DATA_DIR + '/obj365/annotations/objects365_test.json',
    },
    'coco_seg_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
        _FIELDS:
            {'seg_json': _DATA_DIR + '/coco/annotations/stuff_train2017.json',
             'seg_root': _DATA_DIR + '/coco/images/stuffthingmaps/train2017'}
    },
    'coco_seg_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
        _FIELDS:
            {'seg_json': _DATA_DIR + '/coco/annotations/stuff_val2017.json',
             'seg_root': _DATA_DIR + '/coco/images/stuffthingmaps/val2017'}
    },
    'coco_panoptic_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
        _FIELDS:
            {'seg_json': _DATA_DIR + '/coco/annotations/panoptic_train2017.json',
             'seg_root': _DATA_DIR + '/coco/annotations/panoptic_train2017',
             'ignore_label': 255}
    },
    'coco_panoptic_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
        _FIELDS:
            {'seg_json': _DATA_DIR + '/coco/annotations/panoptic_val2017.json',
             'seg_root': _DATA_DIR + '/coco/annotations/panoptic_val2017',
             'ignore_label': 255}
    },
    'ade2017_sceneparsing_train': {
        _IM_DIR:
            _DATA_DIR + '/ADE2017/images/training',
        _ANN_FN:
            _DATA_DIR + '/ADE2017/Json_Annos/ade2017_sceneparsing_train.json',
        _FIELDS:
            {'seg_root': _DATA_DIR + '/ADE2017/annotations_sceneparsing/training',
             'label_shift': -1,
             'ignore_label': 255}
    },
    'ade2017_sceneparsing_val': {
        _IM_DIR:
            _DATA_DIR + '/ADE2017/images/validation',
        _ANN_FN:
            _DATA_DIR + '/ADE2017/Json_Annos/ade2017_sceneparsing_val.json',
        _FIELDS:
            {'seg_root': _DATA_DIR + '/ADE2017/annotations_sceneparsing/validation',
             'label_shift': -1,
             'ignore_label': 255}
    },
    'cihp_semseg_train': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/train_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_train.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/CIHP/train_seg',
             'label_shift': 0}
    },
    'cihp_semseg_val': {
        _IM_DIR:
            _DATA_DIR + '/CIHP/val_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19]),
             'ignore_label': 255,
             'seg_root': _DATA_DIR + '/CIHP/val_seg',
             'label_shift': 0}
    }
}
