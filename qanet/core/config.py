import os.path as osp
from yacs.config import CfgNode as CN


_C = CN()

# ---------------------------------------------------------------------------- #
# Misc options
# --------------------------------------------------------------------------- #
# Version of QANet
_C.VERSION = '0.6'

# Device for training or testing
# E.g., 'cuda' for using GPU, 'cpu' for using CPU
_C.DEVICE = 'cuda'

# Enable cudnn_benchmark for speeding up training
_C.CUDNN = True

# Number of GPUs to use (In fact, this parameter is no longer used.)
_C.NUM_GPUS = 1

# Pixel mean values (BGR order) as a list
_C.PIXEL_MEANS = [102.9801, 115.9465, 122.7717]

# Pixel std values (BGR order) as a list
_C.PIXEL_STDS = [1.0, 1.0, 1.0]

# Clean up the generated files during model testing
_C.CLEAN_UP = True

# Set True to enable model analysis
_C.MODEL_ANALYSE = True

# Directory for saving checkpoints and loggers
_C.CKPT = 'ckpts/CIHP/QANet/QANet_R-50c_512x384_1x/QANet_R-50c_512x384_1x.yaml'

# Display the log per iteration
_C.DISPLAY_ITER = 20

# Root directory of project
_C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
_C.DATA_DIR = osp.abspath(osp.join(_C.ROOT_DIR, 'data'))

# A very small number that's used many times
_C.EPS = 1e-14

# Convert image to RGB format (for QANet pre-trained models), in range 0-1
# image format
# "bgr255": BGR in range 0-255
# "rgb255": RGB in range 0-255
# "bgr": BGR in range 0-1
# "rgb": RGB in range 0-1
_C.IMAGE_FORMAT = "rgb"


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()

# FPN is enabled if True
_C.MODEL.FPN_ON = False

# Indicates the model makes mask segmentation predictor
_C.MODEL.MASK_ON = False

# Indicates the model makes keypoint estimation predictor
_C.MODEL.KEYPOINT_ON = False

# Indicates the model makes parsing predictor
_C.MODEL.PARSING_ON = False

# Indicates the model makes UV predictor
_C.MODEL.UV_ON = False


# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# Type of the optimizer
# E.g., 'SGD', 'RMSPROP', 'ADAM' ...
_C.SOLVER.OPTIMIZER = 'SGD'

# Base learning rate for the specified schedule.
# 0.004 for 32 batch size with warmup   # TO CHECK
# 0.001 for 32 batch size without warmup
_C.SOLVER.BASE_LR = 0.001

# The number of max epochs
_C.SOLVER.MAX_EPOCHS = 140

# Momentum to use with SGD
_C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
_C.SOLVER.WEIGHT_DECAY = 0.0001

# L2 regularization hyperparameter for all Normalization parameters
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

# Adjust the parameters for bias
# TODO: whether keep same with R-CNN
_C.SOLVER.BIAS_LR_FACTOR = 2.0
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

# Multiple learning rate for fine-tuning
# Random initial layer learning rate is LR_MULTIPLE * BASE_LR
_C.SOLVER.LR_MULTIPLE = 1.0  # TODO

# Warm up to SOLVER.BASE_LR over this number of SGD epochs
_C.SOLVER.WARM_UP_EPOCH = 5

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
_C.SOLVER.WARM_UP_FACTOR = 1.0 / 10.0

# WARM_UP_METHOD can be either 'CONSTANT' or 'LINEAR' (i.e., gradual)
_C.SOLVER.WARM_UP_METHOD = 'LINEAR'

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'POLY', 'STEP', 'COSINE', ...
_C.SOLVER.LR_POLICY = 'COSINE'

# For 'POLY', the power in poly to drop LR
_C.SOLVER.LR_POW = 0.9

# For 'STEP', Non-uniform step iterations
_C.SOLVER.STEPS = [50, 75, 90]

# For 'STEP', the current LR is multiplied by SOLVER.GAMMA at each step
_C.SOLVER.GAMMA = 0.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
_C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1

# Snapshot (model checkpoint) period
_C.SOLVER.SNAPSHOT_EPOCHS = -1

# ---------------------------------------------------------------------------- #
# Automatic mixed precision options (during training)
# ---------------------------------------------------------------------------- #
_C.SOLVER.AMP = CN()

_C.SOLVER.AMP.ENABLED = False


# -----------------------------------------------------------------------------
# DataLoader options
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()

# Type of training sampler, default: 'DistributedSampler'
# E.g., 'DistributedSampler', 'RepeatFactorInstanceTrainingSampler', ...
_C.DATALOADER.SAMPLER_TRAIN = "DistributedSampler"

# ---------------------------------------------------------------------------- #
# RepeatFactorTrainingSampler options
# ---------------------------------------------------------------------------- #
_C.DATALOADER.RFTSAMPLER = CN()

# parameters for RepeatFactorTrainingSampler
# rep_times = max(MIN_REPEAT_TIMES, min(MAX_REPEAT_TIMES, math.pow((REPEAT_THRESHOLD / cat_freq),POW)))
_C.DATALOADER.RFTSAMPLER.REPEAT_THRESHOLD = 0.00008
_C.DATALOADER.RFTSAMPLER.POW = 0.5
_C.DATALOADER.RFTSAMPLER.MAX_REPEAT_TIMES = 10000.0
_C.DATALOADER.RFTSAMPLER.MIN_REPEAT_TIMES = 1.0

# ---------------------------------------------------------------------------- #
# Ground truth format
# ---------------------------------------------------------------------------- #
_C.DATALOADER.GT_FORMAT = CN()
# Input mask target format ("poly": polygon, "mask": bitmask)
# "poly" will be loaded on 'cpu', "mask" will be load on 'gpu'
_C.DATALOADER.GT_FORMAT.MASK = "poly"

# Input semseg/parsing target format ("poly": polygon, "mask": picture)
_C.DATALOADER.GT_FORMAT.SEMSEG = "mask"


# ---------------------------------------------------------------------------- #
# Train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# Initialize network with weights from this .pth file
_C.TRAIN.WEIGHTS = ''

# Datasets to train on
# If multiple datasets are listed, the model is trained on their union
_C.TRAIN.DATASETS = ()

# Scales to use during training
_C.TRAIN.SCALES = ([192, 256],)

# Max pixel size of the longest side of a scaled input image
_C.TRAIN.MAX_SIZE = -1

# Image affine type ('cv2', 'roi_align')
_C.TRAIN.AFFINE_MODE = 'cv2'

# Number of Python threads to use for the data loader during training
_C.TRAIN.LOADER_THREADS = 4

# Mini-batch size for training
_C.TRAIN.BATCH_SIZE = 256

# Iteration size for each epoch
_C.TRAIN.ITER_PER_EPOCH = -1

# Use scaled image during training?
_C.TRAIN.SCALE_FACTOR = 0.3

# Use rotated image during training?
_C.TRAIN.ROT_FACTOR = 40

# Use horizontally-flipped images during training?
_C.TRAIN.USE_FLIPPED = True

# Use half body data augmentation during training?
_C.TRAIN.USE_HALF_BODY = False

# Probability of using half body aug
_C.TRAIN.PRO_HALF_BODY = 0.3

# X of using half body aug
_C.TRAIN.X_EXT_HALF_BODY = 0.6

# Y of using half body aug
_C.TRAIN.Y_EXT_HALF_BODY = 0.8

# Num parts for using half body aug
_C.TRAIN.NUM_HALF_BODY = 3

# Upper body ids for half body aug
# (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for COCO keypoint
# (1, 2, 3, 4, 5, 6, 7, 10, 11, 13, 14, 15) for CIHP parsing
_C.TRAIN.UPPER_BODY_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

_C.TRAIN.SELECT_DATA = False   # TODO

_C.TRAIN.CALC_ACC = False   # TODO

# Training will resume from the latest snapshot (model checkpoint) found in the
# output directory
_C.TRAIN.AUTO_RESUME = True

# Save training metrics as json file
_C.TRAIN.SAVE_AS_JSON = False

# Use tensorboard to show training progress
_C.TRAIN.USE_TENSORBOARD = False


# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()

# Initialize network with weights from this .pth file
_C.TEST.WEIGHTS = ""

# Number of Python threads to use for the data loader during training
_C.TEST.LOADER_THREADS = 4

# Use CPU perform data normalization
_C.TEST.USE_CPU_NORM = True

# Datasets to test on
# If multiple datasets are listed, the model is trained on their union
_C.TEST.DATASETS = ()

# Scales to use during testing
_C.TEST.SCALE = [192, 256]

# Max pixel size of the longest side of a scaled input image
_C.TEST.MAX_SIZE = -1

# Image affine type ('cv2', 'roi_align')
_C.TEST.AFFINE_MODE = 'roi_align'

# Number of images in each GPU for testing
_C.TEST.IMS_PER_GPU = 1

# Number of instances in each batch for testing
_C.TEST.INSTANCES_PER_BATCH = -1

# Frequency for testing during training
# -1 means never testing during training
# 1 means testing after every training epoch
# SOLVER.MAX_EPOCHS means testing after entire training epoch
_C.TEST.FREQUENCY_EPOCHS = -1

# Instance score threshold for keypoint
_C.TEST.IMAGE_THRESH = 0.0

# detection results of instance for testing
#  "" means using GT bbox
_C.TEST.INSTANCE_BBOX_FILE = ""

# ---------------------------------------------------------------------------- #
# Precise BN
# ---------------------------------------------------------------------------- #
_C.TEST.PRECISE_BN = CN()

_C.TEST.PRECISE_BN.ENABLED = False

# Set same with test period while test during training.
_C.TEST.PRECISE_BN.PERIOD = 0

_C.TEST.PRECISE_BN.NUM_ITER = 200

# ---------------------------------------------------------------------------- #
# Data augmentation options (during testing)
# ---------------------------------------------------------------------------- #
# Details in https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
_C.TEST.AUG = CN()

# Horizontally flip during testing
_C.TEST.AUG.H_FLIP = False

# Scale image during testing
_C.TEST.AUG.SCALES = ()

_C.TEST.AUG.POST_PROCESS = True

_C.TEST.AUG.SHIFT_HEATMAP = False


# ---------------------------------------------------------------------------- #
# Backbone options
# (CrossNet, FBNet, MNasNet, MobileNet_v1, MobileNet_v2, PeleeNet,
#  ProxylessNas, ResNet, ResNeXt, ShuffleNet_v2, VGG)
# ---------------------------------------------------------------------------- #
_C.BACKBONE = CN()

# The backbone conv body to use
_C.BACKBONE.CONV_BODY = 'resnet'

# The eps of batch_norm layer
_C.BACKBONE.BN_EPS = 1e-5

# ---------------------------------------------------------------------------- #
# HRNet options
# ---------------------------------------------------------------------------- #
_C.BACKBONE.HRNET = CN()

# Network initial width
_C.BACKBONE.HRNET.WIDTH = 18

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
_C.BACKBONE.HRNET.AVG_DOWN = False

# Use a global feature in each stage
_C.BACKBONE.HRNET.USE_GLOBAL = False

# Use SplAtConv2d in bottleneck.
# 2 for resnest
_C.BACKBONE.HRNET.RADIX = 1

# Type of 3x3 convolution layer in each block
# E.g., 'Conv2d', 'Conv2dWS', 'DeformConv', 'MDeformConv', ...
_C.BACKBONE.HRNET.STAGE_WITH_CONV = ('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d')

# # Type of normalization
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.BACKBONE.HRNET.NORM = 'BN'

# Type of context module in each block
# E.g., 'SE', 'GCB', ...
_C.BACKBONE.HRNET.STAGE_WITH_CTX = ("", "", "", "")

# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
_C.BACKBONE.RESNET = CN()

# The number of layers in each block
# (2, 2, 2, 2) for resnet18 with basicblock
# (3, 4, 6, 3) for resnet34 with basicblock
# (3, 4, 6, 3) for resnet50
# (3, 4, 23, 3) for resnet101
# (3, 8, 36, 3) for resnet152
_C.BACKBONE.RESNET.LAYERS = (3, 4, 6, 3)

# Network stem width
_C.BACKBONE.RESNET.STEM_WIDTH = 64

# Network initial width
_C.BACKBONE.RESNET.WIDTH = 64

# Use bottleneck block, False for basicblock
_C.BACKBONE.RESNET.BOTTLENECK = True

# Use a aligned module in each block
_C.BACKBONE.RESNET.USE_ALIGN = False

# Use weight standardization
_C.BACKBONE.RESNET.USE_WS = False

# Place the stride 2 conv on the 3x3 filter.
# True for resnet-b
_C.BACKBONE.RESNET.STRIDE_3X3 = False

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnet-c
_C.BACKBONE.RESNET.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnet-d
_C.BACKBONE.RESNET.AVG_DOWN = False

# Use SplAtConv2d in bottleneck.
# 2 for resnest
_C.BACKBONE.RESNET.RADIX = 1

# Type of 3x3 convolution layer in each block
# E.g., 'Conv2d', 'Conv2dWS', 'DeformConv', 'MDeformConv', ...
_C.BACKBONE.RESNET.STAGE_WITH_CONV = ('Conv2d', 'Conv2d', 'Conv2d', 'Conv2d')

# # Type of normalization
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.BACKBONE.RESNET.NORM = 'BN'

# Type of context module in each block
# E.g., 'SE', 'GCB', ...
_C.BACKBONE.RESNET.STAGE_WITH_CTX = ("", "", "", "")

# Network output stride, 32:c5, 16:c4, 8:c3
_C.BACKBONE.RESNET.STRIDE = 32


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.FPN = CN()

# The Body of FPN to use
# (e.g., "pfpn", "xdeconv")
_C.FPN.BODY = "xdeconv"

# Use C5 or P5 to generate P6
_C.FPN.USE_C5 = True

# Channel dimension of the FPN feature levels
_C.FPN.DIM = 256

# FPN may be used for just RPN, just object detection, or both
# E.g., "conv2"-like level
_C.FPN.LOWEST_BACKBONE_LVL = 2

# E.g., "conv5"-like level
_C.FPN.HIGHEST_BACKBONE_LVL = 5

# Use extra FPN levels, as done in the RetinaNet paper
_C.FPN.EXTRA_CONV_LEVELS = False

# Type of normalization in the FPN-specific layers (lateral, etc.)
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.FPN.NORM = ""

# Use Weight Standardization in the FPN-specific layers (lateral, etc.)
_C.FPN.USE_WS = False

# ---------------------------------------------------------------------------- #
# FPN panoptic head options
# ---------------------------------------------------------------------------- #
_C.FPN.PANOPTIC = CN()

# panoptic head conv dim out
_C.FPN.PANOPTIC.CONV_DIM = 256

# Use FPN module before the panoptic head
_C.FPN.PANOPTIC.USE_FPN = True

# Type of normalization in the NASFPN layers
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.FPN.PANOPTIC.NORM = ""

# ---------------------------------------------------------------------------- #
# FPN deconvx body options
# ---------------------------------------------------------------------------- #
_C.FPN.DECONVX = CN()

# Number of ConvTranspose channels in the deconvx body
_C.FPN.DECONVX.HEAD_DIM = 256

# Type of normalization in the Semantic head
# E.g., "FrozenBN", "BN", "SyncBN", "GN", "MixBN", "MixGN", ...
_C.FPN.DECONVX.NORM = "BN"

# Decay factor of head ConvTranspose channels
_C.FPN.DECONVX.HEAD_DECAY_FACTOR = 1

# Size of the kernels to use in all ConvTranspose operations
_C.FPN.DECONVX.HEAD_KERNEL = 4

# Number of stacked ConvTranspose layers in deconvx body
_C.FPN.DECONVX.NUM_DECONVS = 3

# Use bias in ConvTranspose layer
_C.FPN.DECONVX.WITH_BIAS = False


# ---------------------------------------------------------------------------- #
# Mask options
# ---------------------------------------------------------------------------- #
_C.MASK = CN()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
_C.MASK.MASK_HEAD = 'simple_none_head'

# Output module of mask head
_C.MASK.MASK_OUTPUT = 'conv1x1_outputs'

# Output module of mask loss
_C.MASK.MASK_LOSS = 'mask_loss'

# Mask Number for mask estimation
_C.MASK.NUM_CLASSES = 81

# Loss weight for mask
_C.MASK.LOSS_WEIGHT = 1.0

# Use Mask IoU for mask head
_C.MASK.MASKIOU_ON = False

# Weights for calculating quality score (bbox_scores, iou_scores, pixel_scores)
_C.MASK.QUALITY_WEIGHTS = (1.0, 1.0, 0.0)

# Threshold of mask prob to calculate pixel score
_C.MASK.PIXEL_SCORE_TH = 0.25

# ---------------------------------------------------------------------------- #
# Mask gce head options
# ---------------------------------------------------------------------------- #
_C.MASK.GCE_HEAD = CN()

# Hidden Conv layer dimension
_C.MASK.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPP
_C.MASK.GCE_HEAD.ASPP_DIM = 256

# Dilation for ASPP
_C.MASK.GCE_HEAD.ASPP_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before
_C.MASK.GCE_HEAD.NUM_CONVS_BEFORE_ASPP = 0

# Number of stacked Conv layers in GCE head after
_C.MASK.GCE_HEAD.NUM_CONVS_AFTER_ASPP = 0

# Use NonLocal in the Keypoint gce head
_C.MASK.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
_C.MASK.GCE_HEAD.NL_RATIO = 1.0

# Type of normalization in the PARSING gce head
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.MASK.GCE_HEAD.NORM = ""

# ---------------------------------------------------------------------------- #
# Mask IoU options
# ---------------------------------------------------------------------------- #
_C.MASK.MASKIOU = CN()

# The head of Mask IoU to use
# (e.g., "convx_head")
_C.MASK.MASKIOU.MASKIOU_HEAD = "maskiou_head"

# Output module of Mask IoU head
_C.MASK.MASKIOU.MASKIOU_OUTPUT = "maskiou_output"

# Number of stacked Conv layers in Mask IoU head
_C.MASK.MASKIOU.NUM_CONVS = 2

# Hidden Conv layer dimension of Mask IoU head
_C.MASK.MASKIOU.CONV_DIM = 512

# Type of normalization in the MASK IoU head
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.MASK.MASKIOU.NORM = ""

# Loss weight for Mask IoU head
_C.MASK.MASKIOU.LOSS_WEIGHT = 1.0


# ---------------------------------------------------------------------------- #
# Keypoint options
# ---------------------------------------------------------------------------- #
_C.KEYPOINT = CN()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
_C.KEYPOINT.KEYPOINT_HEAD = 'simple_none_head'

# Output module of keypoint head
_C.KEYPOINT.KEYPOINT_OUTPUT = 'conv1x1_outputs'

# Use target weight during training
_C.KEYPOINT.USE_TARGET_WEIGHT = True

# Keypoints Number for keypoint estimation
_C.KEYPOINT.NUM_KEYPOINTS = 17

# Soft target type
_C.KEYPOINT.TARGET_TYPE = 'gaussian'

# Usually 1 / 4. size of input
_C.KEYPOINT.PROB_SIZE = (48, 64)

# Sigma
_C.KEYPOINT.SIGMA = 2

# OKS score threshold for testing
_C.KEYPOINT.OKS_THRESH = 0.9

# Index thresh
_C.KEYPOINT.INDEX_THRESH = 0.2

# Loss weight for keypoint
_C.KEYPOINT.LOSS_WEIGHT = 1.0

# Weights for calculating quality score (bbox_scores, iou_scores, pixel_scores)
_C.KEYPOINT.QUALITY_WEIGHTS = (1.0, 1.0, 0.0)

# ---------------------------------------------------------------------------- #
# Keypoint gce head options
# ---------------------------------------------------------------------------- #
_C.KEYPOINT.GCE_HEAD = CN()

# Hidden Conv layer dimension
_C.KEYPOINT.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPP
_C.KEYPOINT.GCE_HEAD.ASPP_DIM = 256

# Dilation for ASPP
_C.KEYPOINT.GCE_HEAD.ASPP_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before 
_C.KEYPOINT.GCE_HEAD.NUM_CONVS_BEFORE_ASPP = 0

# Number of stacked Conv layers in GCE head after 
_C.KEYPOINT.GCE_HEAD.NUM_CONVS_AFTER_ASPP = 0

# Use NonLocal in the Keypoint gce head
_C.KEYPOINT.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
_C.KEYPOINT.GCE_HEAD.NL_RATIO = 1.0

# Type of normalization in the KEYPOINT gce head
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.KEYPOINT.GCE_HEAD.NORM = ""


# ---------------------------------------------------------------------------- #
# Parsing options
# ---------------------------------------------------------------------------- #
_C.PARSING = CN()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
_C.PARSING.PARSING_HEAD = 'simple_none_head'

# Output module of parsing head
_C.PARSING.PARSING_OUTPUT = 'conv1x1_outputs'

# Output module of parsing loss
_C.PARSING.PARSING_LOSS = 'parsing_loss'

# Parsing Number for parsing estimation
_C.PARSING.NUM_PARSING = 20

# Loss weight for parsing
_C.PARSING.LOSS_WEIGHT = 1.0

# Lovasz loss weight for parsing
_C.PARSING.LOVASZ_LOSS_WEIGHT = 0.0

# Use Parsing IoU for Parsing head
_C.PARSING.PARSINGIOU_ON = False

# Use Quality for Parsing head
_C.PARSING.QUALITY_ON = False

# Parsing evaluating metrics to use
# (e.g., "mIoU", "APp", "APr")
_C.PARSING.METRICS = ['mIoU', 'APp', 'APr']

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision parsings
_C.PARSING.SCORE_THRESH = 0.01

# Weights for calculating quality score (bbox_scores, iou_scores, pixel_scores)
_C.PARSING.QUALITY_WEIGHTS = (1.0, 1.0, 0.0)

# Threshold of parsing prob to calculate pixel score
_C.PARSING.PIXEL_SCORE_TH = 0.2

# Minimum score threshold (assuming scores in a [0, 1] range) for semantice
# segmentation results.
# 0.3 for CIHP, 0.05 for MHP-v2
_C.PARSING.SEMSEG_SCORE_THRESH = 0.3

# Overlap threshold used for parsing non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.PARSING.PARSING_NMS_TH = 0.6

# ---------------------------------------------------------------------------- #
# Parsing gce head options
# ---------------------------------------------------------------------------- #
_C.PARSING.GCE_HEAD = CN()

# Hidden Conv layer dimension
_C.PARSING.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPP
_C.PARSING.GCE_HEAD.ASPP_DIM = 256

# Dilation for ASPP
_C.PARSING.GCE_HEAD.ASPP_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before 
_C.PARSING.GCE_HEAD.NUM_CONVS_BEFORE_ASPP = 0

# Number of stacked Conv layers in GCE head after 
_C.PARSING.GCE_HEAD.NUM_CONVS_AFTER_ASPP = 0

# Use NonLocal in the Keypoint gce head
_C.PARSING.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
_C.PARSING.GCE_HEAD.NL_RATIO = 1.0

# Type of normalization in the PARSING gce head
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.PARSING.GCE_HEAD.NORM = ""

# ---------------------------------------------------------------------------- #
# Parsing IoU options
# ---------------------------------------------------------------------------- #
_C.PARSING.PARSINGIOU = CN()

# The head of Parsing IoU to use
# (e.g., "parsingiou_head")
_C.PARSING.PARSINGIOU.PARSINGIOU_HEAD = "parsingiou_head"

# Output module of Parsing IoU head
_C.PARSING.PARSINGIOU.PARSINGIOU_OUTPUT = "parsingiou_output"

# Number of stacked Conv layers in Parsing IoU head
_C.PARSING.PARSINGIOU.NUM_CONVS = 2

# Hidden Conv layer dimension of Parsing IoU head
_C.PARSING.PARSINGIOU.CONV_DIM = 512

# Type of normalization in the PARSING IoU head
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.PARSING.PARSINGIOU.NORM = ""

# Loss weight for Parsing IoU head
_C.PARSING.PARSINGIOU.LOSS_WEIGHT = 1.0


# ---------------------------------------------------------------------------- #
# UV options
# ---------------------------------------------------------------------------- #
_C.UV = CN()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
_C.UV.UV_HEAD = 'simple_none_head'

# Output module of UV head
_C.UV.UV_OUTPUT = 'UV_outputs'

# Output module of UV loss
_C.UV.UV_LOSS = 'UV_loss'

# Number of parts in the dataset
_C.UV.NUM_PARTS = 14

# Number of patches in the dataset
_C.UV.NUM_PATCHES = 24

# Weights
_C.UV.INDEX_WEIGHTS = 5.0
_C.UV.PART_WEIGHTS = 1.0
_C.UV.POINT_REGRESSION_WEIGHTS = 0.001

# Weights for calculating quality score (bbox_scores, iou_scores, pixel_scores)
_C.UV.QUALITY_WEIGHTS = (1.0, 1.0, 0.0)

# Index thresh
_C.UV.INDEX_THRESH = 0.9

# UV evaluating calc_mode to use
# (e.g., "GPSm", "GPS", "IOU")
_C.UV.CALC_MODE = "GPSm"

# ---------------------------------------------------------------------------- #
# UV gce head options
# ---------------------------------------------------------------------------- #
_C.UV.GCE_HEAD = CN()

# Hidden Conv layer dimension
_C.UV.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPP
_C.UV.GCE_HEAD.ASPP_DIM = 256

# Dilation for ASPP
_C.UV.GCE_HEAD.ASPP_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before 
_C.UV.GCE_HEAD.NUM_CONVS_BEFORE_ASPP = 0

# Number of stacked Conv layers in GCE head after 
_C.UV.GCE_HEAD.NUM_CONVS_AFTER_ASPP = 0

# Use NonLocal in the UV gce head
_C.UV.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
_C.UV.GCE_HEAD.NL_RATIO = 1.0

# Type of normalization in the UV gce head
# E.g., 'FrozenBN', 'BN', 'SyncBN', 'GN', 'MixBN', 'MixGN', ...
_C.UV.GCE_HEAD.NORM = ""


# ---------------------------------------------------------------------------- #
# Visualization options
# ---------------------------------------------------------------------------- #
_C.VIS = CN()

# Dump detection visualizations
_C.VIS.ENABLED = False

# Score threshold for visualization
_C.VIS.VIS_TH = 0.9

# ---------------------------------------------------------------------------- #
# Show box options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_BOX = CN()

# Visualizing detection bboxes
_C.VIS.SHOW_BOX.ENABLED = True

# Visualization color scheme
# 'green', 'category' or 'instance'
_C.VIS.SHOW_BOX.COLOR_SCHEME = 'green'

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
_C.VIS.SHOW_BOX.COLORMAP = 'COCO81'

# Border thick
_C.VIS.SHOW_BOX.BORDER_THICK = 2

# ---------------------------------------------------------------------------- #
# Show class options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_CLASS = CN()

# Visualizing detection classes
_C.VIS.SHOW_CLASS.ENABLED = True

# Default: gray
_C.VIS.SHOW_CLASS.COLOR = (218, 227, 218)

# Font scale of class string
_C.VIS.SHOW_CLASS.FONT_SCALE = 0.45

# ---------------------------------------------------------------------------- #
# Show Mask options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_MASK = CN()

# Visualizing detection classes
_C.VIS.SHOW_MASK.ENABLED = True

# False = (255, 255, 255) = white
_C.VIS.SHOW_MASK.MASK_COLOR_FOLLOW_BOX = True

# Mask ahpha
_C.VIS.SHOW_MASK.MASK_ALPHA = 0.4

# Whether show border
_C.VIS.SHOW_MASK.SHOW_BORDER = True

# Border color, (255, 255, 255) for white, (0, 0, 0) for black
_C.VIS.SHOW_MASK.BORDER_COLOR = (255, 255, 255)

# Border thick
_C.VIS.SHOW_MASK.BORDER_THICK = 2

# ---------------------------------------------------------------------------- #
# Show keypoints options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_KPS = CN()

# Visualizing detection keypoints
_C.VIS.SHOW_KPS.ENABLED = True

# Keypoints threshold
_C.VIS.SHOW_KPS.KPS_TH = 0.4

# Default: white
_C.VIS.SHOW_KPS.KPS_COLOR_WITH_PARSING = (255, 255, 255)

# Keypoints alpha
_C.VIS.SHOW_KPS.KPS_ALPHA = 0.7

# Link thick
_C.VIS.SHOW_KPS.LINK_THICK = 2

# Circle radius
_C.VIS.SHOW_KPS.CIRCLE_RADIUS = 3

# Circle thick
_C.VIS.SHOW_KPS.CIRCLE_THICK = -1

# ---------------------------------------------------------------------------- #
# Show parsing options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_PARSS = CN()

# Visualizing detection classes
_C.VIS.SHOW_PARSS.ENABLED = True

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
_C.VIS.SHOW_PARSS.COLORMAP = 'CIHP20'

# Parsing alpha
_C.VIS.SHOW_PARSS.PARSING_ALPHA = 0.4

# Whether show border
_C.VIS.SHOW_PARSS.SHOW_BORDER = True

# Border color
_C.VIS.SHOW_PARSS.BORDER_COLOR = (255, 255, 255)

# Border thick
_C.VIS.SHOW_PARSS.BORDER_THICK = 1

# ---------------------------------------------------------------------------- #
# Show uv options
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_UV = CN()

# Visualizing detection classes
_C.VIS.SHOW_UV.ENABLED = True

# Whether show border
_C.VIS.SHOW_UV.SHOW_BORDER = True

# Border thick
_C.VIS.SHOW_UV.BORDER_THICK = 6

# Grid thick
_C.VIS.SHOW_UV.GRID_THICK = 2

# Grid lines num
_C.VIS.SHOW_UV.LINES_NUM = 15

# ---------------------------------------------------------------------------- #
# Show hier options (Not implemented)
# ---------------------------------------------------------------------------- #
_C.VIS.SHOW_HIER = CN()

# Visualizing detection classes
_C.VIS.SHOW_HIER.ENABLED = True

# Border thick
_C.VIS.SHOW_HIER.BORDER_THICK = 2


def get_cfg():

  return _C.clone()


def infer_cfg(cfg):
    pass
    # if cfg.MODEL.MASK_ON or cfg.MODEL.EMBED_MASK_ON or cfg.MODEL.SOLO_ON:
    #     cfg.MODEL.HAS_MASK = True

    return cfg
