import os.path as osp

from lib.datasets.dataset_catalog import COMMON_DATASETS

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'
_FIELDS = 'extra_fields'

# Available datasets
_DATASETS = {
    'ATR_train': {
        _IM_DIR:
            _DATA_DIR + '/ATR/train_img',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_train.json',
        _FIELDS:
            {'flip_map': ([9, 10], [12, 13], [14, 15])},
    },
    'ATR_val': {
        _IM_DIR:
            _DATA_DIR + '/ATR/val_img',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_val.json',
        _FIELDS:
            {'flip_map': ([9, 10], [12, 13], [14, 15])},
    },
    'LIP_train': {
        _IM_DIR:
            _DATA_DIR + '/LIP/train_img',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_train.json',
        _FIELDS:
            {'flip_map': ([9, 10], [12, 13], [14, 15])},
    },
    'LIP_val': {
        _IM_DIR:
            _DATA_DIR + '/LIP/val_img',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_val.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19])},
    },
    'LIP_test': {
        _IM_DIR:
            _DATA_DIR + '/LIP/val_img',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_test.json',
        _FIELDS:
            {'flip_map': ([14, 15], [16, 17], [18, 19])},
    }
}
_DATASETS.update(COMMON_DATASETS)


def datasets():
    """Retrieve the list of available dataset names."""
    return _DATASETS.keys()


def contains(name):
    """Determine if the dataset is in the catalog."""
    return name in _DATASETS.keys()


def get_im_dir(name):
    """Retrieve the image directory for the dataset."""
    return _DATASETS[name][_IM_DIR]


def get_ann_fn(name):
    """Retrieve the annotation file for the dataset."""
    return _DATASETS[name][_ANN_FN]


def get_extra_fields(name):
    """Retrieve the extra fields for the dataset."""
    if _FIELDS in _DATASETS[name]:
        return _DATASETS[name][_FIELDS]
    else:
        return {}
