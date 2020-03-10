import os.path as osp

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

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
            _DATA_DIR + '/CIHP/train_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_train.json',
    },
    'CIHP_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/val_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
    },
    'CIHP_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/test_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_test.json',
    },
    'MHP-v2_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/train_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_train.json',
    },
    'MHP-v2_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/val_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_val.json',
    },
    'MHP-v2_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_all.json',
    },
    'MHP-v2_test_inter_top10': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_inter_top10.json',
    },
    'MHP-v2_test_inter_top20': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/MHP-v2/annotations/MHP-v2_test_inter_top20.json',
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
    },
    'lvis_v0.5_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_val_2017.json',
    },
    'lvis_v0.5_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/lvis/lvis_v0.5_image_info_test.json',
    }
}
