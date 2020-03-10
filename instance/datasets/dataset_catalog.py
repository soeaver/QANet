import os.path as osp

from pet.utils.data.dataset_catalog import COMMON_DATASETS

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'data'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Available datasets
_DATASETS = {
    'ATR_train': {
        _IM_DIR:
            _DATA_DIR + '/ATR/train_img',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_train.json',
    },
    'ATR_val': {
        _IM_DIR:
            _DATA_DIR + '/ATR/val_img',
        _ANN_FN:
            _DATA_DIR + '/ATR/annotations/ATR_val.json',
    },
    'LIP_train': {
        _IM_DIR:
            _DATA_DIR + '/LIP/train_img',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_train.json',
    },
    'LIP_val': {
        _IM_DIR:
            _DATA_DIR + '/LIP/val_img',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_val.json',
    },
    'LIP_test': {
        _IM_DIR:
            _DATA_DIR + '/LIP/val_img',
        _ANN_FN:
            _DATA_DIR + '/LIP/annotations/LIP_test.json',
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
