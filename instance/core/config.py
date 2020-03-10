import os
import os.path as osp
import copy
import yaml
import numpy as np
from ast import literal_eval

from pet.utils.collections import AttrDict

__C = AttrDict()
cfg = __C


# ---------------------------------------------------------------------------- #
# Misc options
# --------------------------------------------------------------------------- #
# Device for training or testing
# E.g., 'cuda' for using GPU, 'cpu' for using CPU
__C.DEVICE = 'cuda'

# Number of GPUs to use (In fact, this parameter is no longer used.)
__C.NUM_GPUS = 1

# Pixel mean values (BGR order) as a list
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# Pixel std values (BGR order) as a list
__C.PIXEL_STDS = np.array([[[1.0, 1.0, 1.0]]])

# Clean up the generated files during model testing
__C.CLEAN_UP = True

# Calculation the model flops and params
__C.CALC_FLOPS = True

# Directory for saving checkpoints and loggers
__C.CKPT = 'ckpts/instance/mscoco/simple_R-50-1x64d-D3K4C256_256x192_adam_1x'

# Display the log per iteration
__C.DISPLAY_ITER = 20

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# A very small number that's used many times
__C.EPS = 1e-14

# Convert image to RGB format (for Pet pre-trained models), in range 0-1
__C.TO_RGB = False


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

# The type of model to use
# The string must match a function in the modeling.model_builder module
# (e.g., 'generalized_instance', ...)
__C.MODEL.TYPE = 'generalized_instance'

# FPN is enabled if True
__C.MODEL.FPN_ON = False

# Indicates the model makes mask segmentation predictor
__C.MODEL.MASK_ON = False

# Indicates the model makes keypoint estimation predictor
__C.MODEL.KEYPOINT_ON = False

# Indicates the model makes parsing predictor
__C.MODEL.PARSING_ON = False

# Indicates the model makes UV predictor
__C.MODEL.UV_ON = False

# Type of batch normalizaiton, default: 'freeze'
# E.g., 'normal', 'freeze', 'sync', ...
__C.MODEL.BATCH_NORM = 'normal'


# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# Type of the optimizer
# E.g., 'SGD', 'RMSPROP', 'ADAM' ...
__C.SOLVER.OPTIMIZER = 'SGD'

# Base learning rate for the specified schedule.
# 0.004 for 32 batch size with warmup   # TO CHECK
# 0.001 for 32 batch size without warmup
__C.SOLVER.BASE_LR = 0.001

# The number of max epochs
__C.SOLVER.MAX_EPOCHS = 140

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

# L2 regularization hyperparameter for GroupNorm's parameters
__C.SOLVER.WEIGHT_DECAY_GN = 0.0

# Whether to double the learning rate for bias
__C.SOLVER.BIAS_DOUBLE_LR = False

# Whether to have weight decay on bias as well
__C.SOLVER.BIAS_WEIGHT_DECAY = True

# Multiple learning rate for fine-tuning
# Random initial layer learning rate is LR_MULTIPLE * BASE_LR
__C.SOLVER.LR_MULTIPLE = 1.0  # TODO

# Warm up to SOLVER.BASE_LR over this number of SGD epochs
__C.SOLVER.WARM_UP_EPOCH = 5

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 10.0

# WARM_UP_METHOD can be either 'CONSTANT' or 'LINEAR' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'LINEAR'

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'POLY', 'STEP', 'COSINE', ...
__C.SOLVER.LR_POLICY = 'COSINE'

# For 'POLY', the power in poly to drop LR
__C.SOLVER.LR_POW = 0.9

# For 'STEP', Non-uniform step iterations
__C.SOLVER.STEPS = [50, 75, 90]

# For 'STEP', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1

# Snapshot (model checkpoint) period
__C.SOLVER.SNAPSHOT_EPOCHS = -1

# ---------------------------------------------------------------------------- #
# Automatic mixed precision options (during training)
# ---------------------------------------------------------------------------- #
__C.SOLVER.AMP = AttrDict()

__C.SOLVER.AMP.ENABLED = False

# Opt level for amp initialize
# (e.g., 'O0 = fp32 training', 'O1 = conservative mixed precision training',
#  'O2 = fast mixed precision', 'O3 = fp16 training')
__C.SOLVER.AMP.OPT_LEVEL = 'O2'

# Batchnorm op uses fp32 training
__C.SOLVER.AMP.KEEP_BN_FP32 = True

# Using dynamic loss scaling,
# can be overridden to use static loss scaling, e.g., 128.0
__C.SOLVER.AMP.LOSS_SCALE = 'dynamic'


# ---------------------------------------------------------------------------- #
# Train options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Initialize network with weights from this .pth file
__C.TRAIN.WEIGHTS = 'weights/vgg16-73.4.pth'

# Datasets to train on
# If multiple datasets are listed, the model is trained on their union
__C.TRAIN.DATASETS = ()

# Scales to use during training
__C.TRAIN.SCALES = ([192, 256],)

# Number of Python threads to use for the data loader during training
__C.TRAIN.LOADER_THREADS = 4

# Mini-batch size for training
__C.TRAIN.BATCH_SIZE = 256

# Iteration size for each epoch
__C.TRAIN.ITER_PER_EPOCH = -1

# Use scaled image during training?
__C.TRAIN.SCALE_FACTOR = 0.3

# Use rotated image during training?
__C.TRAIN.ROT_FACTOR = 40

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Use half body data augmentation during training?
__C.TRAIN.USE_HALF_BODY = False

# Probability of using half body aug
__C.TRAIN.PRO_HALF_BODY = 0.3

# X of using half body aug
__C.TRAIN.X_EXT_HALF_BODY = 0.6

# Y of using half body aug
__C.TRAIN.Y_EXT_HALF_BODY = 0.8

# NUM_KEYPOINTS_HALF_BODY of using half body aug
__C.TRAIN.NUM_KEYPOINTS_HALF_BODY = 8

__C.TRAIN.SELECT_DATA = False   # TODO

__C.TRAIN.CALC_ACC = False   # TODO

# Training will resume from the latest snapshot (model checkpoint) found in the
# output directory
__C.TRAIN.AUTO_RESUME = True


# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Initialize network with weights from this .pth file
__C.TEST.WEIGHTS = ''

# Number of Python threads to use for the data loader during training
__C.TEST.LOADER_THREADS = 4

# Use CPU perform data normalization
__C.TEST.USE_CPU_NORM = True

# Datasets to test on
# If multiple datasets are listed, the model is trained on their union
__C.TEST.DATASETS = ()

# Scales to use during testing
__C.TEST.SCALE = [192, 256]

# Number of images in each GPU for testing
__C.TEST.IMS_PER_GPU = 32

# Frequency for testing during training
# -1 means never testing during training
# 1 means testing after every training epoch
# SOLVER.MAX_EPOCHS means testing after entire training epoch
__C.TEST.FREQUENCY_EPOCHS = -1

# Instance score threshold for keypoint
__C.TEST.IMAGE_THRESH = 0.0

# detection results of instance for testing
#  '' means using GT bbox
__C.TEST.INSTANCE_BBOX_FILE = ''

# ---------------------------------------------------------------------------- #
# Data augmentation options (during testing)
# ---------------------------------------------------------------------------- #
# Details in https://github.com/pytorch/vision/blob/master/torchvision/transforms/transforms.py
__C.TEST.AUG = AttrDict()

# Horizontally flip during testing
__C.TEST.AUG.H_FLIP = False

# Scale image during testing
__C.TEST.AUG.SCALES = ()

__C.TEST.AUG.POST_PROCESS = True

__C.TEST.AUG.SHIFT_HEATMAP = False


# ---------------------------------------------------------------------------- #
# Backbone options
# (CrossNet, FBNet, MNasNet, MobileNet_v1, MobileNet_v2, PeleeNet,
#  ProxylessNas, ResNet, ResNeXt, ShuffleNet_v2, VGG)
# ---------------------------------------------------------------------------- #
__C.BACKBONE = AttrDict()

# The backbone conv body to use
__C.BACKBONE.CONV_BODY = 'resnet'

# The eps of batch_norm layer
__C.BACKBONE.BN_EPS = 1e-5

# ---------------------------------------------------------------------------- #
# CBNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.CBNET = AttrDict()

# Number of backbone used for CBNet
# 2 as Dual-Backbone, 3 as Triple-Backbone
__C.BACKBONE.CBNET.NUM_BK = 2

# Use batch normalization in composition module
__C.BACKBONE.CBNET.USE_BN = False

# Use group normalization in composition module
__C.BACKBONE.CBNET.USE_GN = False

# ---------------------------------------------------------------------------- #
# CrossNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.CX = AttrDict()

# The number of layers in each block
__C.BACKBONE.CX.LAYERS = (3, 7, 3)

# Network initial width
__C.BACKBONE.CX.WIDTH = 70

# Expansion coefficient of each block
__C.BACKBONE.CX.EXPANSION = 0.4

# Kernel size of depth-wise separable convolution layers
__C.BACKBONE.CX.KERNEL = 5

# Groups of the last 1x1 convolution layer in each block
__C.BACKBONE.CX.GROUPS = 2

# Depth of CrossPod
__C.BACKBONE.CX.DEPTH = 3

# Use a squeeze-and-excitation module in each block
__C.BACKBONE.CX.USE_SE = False

# ---------------------------------------------------------------------------- #
# EfficientNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.EFFI = AttrDict()

# Network setting of EfficientNet
__C.BACKBONE.EFFI.SETTING = 'b0'

# Use Conv2dSamePadding to replace Conv2d for fitting tf-original implementation
__C.BACKBONE.EFFI.SAME_PAD = True

# ---------------------------------------------------------------------------- #
# FBNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.FB = AttrDict()

# Network setting of FBNet
__C.BACKBONE.FB.SETTING = 'C'

# Network widen factor
__C.BACKBONE.FB.WIDEN_FACTOR = 1.0

# ---------------------------------------------------------------------------- #
# HRNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.HRNET = AttrDict()

# Network initial width
__C.BACKBONE.HRNET.WIDTH = 18

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
__C.BACKBONE.HRNET.AVG_DOWN = False

# Use a squeeze-and-excitation module in each block
__C.BACKBONE.HRNET.USE_SE = False

# Use a global feature in each stage
__C.BACKBONE.HRNET.USE_GLOBAL = False

# Use group normalization
__C.BACKBONE.HRNET.USE_GN = False

# Use a aligned module in each block
__C.BACKBONE.HRNET.USE_ALIGN = False

# Type of 3x3 convolution layer in each block
# 'deform' for dcnv1, 'deformv2' for dcnv2
__C.BACKBONE.HRNET.STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')

# ---------------------------------------------------------------------------- #
# MNasNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.MNAS = AttrDict()

# Network setting of MNasNet
__C.BACKBONE.MNAS.SETTING = 'A'

# Network widen factor
__C.BACKBONE.MNAS.WIDEN_FACTOR = 1.0

# ---------------------------------------------------------------------------- #
# MobileNet V1 options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.MV1 = AttrDict()

# The number of layers in each block
__C.BACKBONE.MV1.LAYERS = (2, 2, 6, 2)

# The initial width of each block
__C.BACKBONE.MV1.NUM_CHANNELS = [32, 64, 128, 256, 512, 1024]

# Kernel size of depth-wise separable convolution layers
__C.BACKBONE.MV1.KERNEL = 3

# Network widen factor
__C.BACKBONE.MV1.WIDEN_FACTOR = 1.0

# Use a squeeze-and-excitation module in each block
__C.BACKBONE.MV1.USE_SE = False

# Use dropblock in C4 and C5
__C.BACKBONE.MV1.USE_DP = False

# ---------------------------------------------------------------------------- #
# MobileNet V2 options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.MV2 = AttrDict()

# Network widen factor
__C.BACKBONE.MV2.WIDEN_FACTOR = 1.0

# Use a squeeze-and-excitation module in each block
__C.BACKBONE.MV2.USE_SE = False

# ---------------------------------------------------------------------------- #
# MobileNet V3 options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.MV3 = AttrDict()

# Network setting of MobileNet V3
__C.BACKBONE.MV3.SETTING = 'large'

# Network widen factor
__C.BACKBONE.MV3.WIDEN_FACTOR = 1.0

# Se module mid channel base, if True use innerplanes, False use inplanes
__C.BACKBONE.MV3.SE_REDUCE_MID = True

# Se module mid channel divisible. This param is to fit otf-fficial implementation
__C.BACKBONE.MV3.SE_DIVISIBLE = False

# Use conv bias in head. This param is to fit tf-official implementation
__C.BACKBONE.MV3.HEAD_USE_BIAS = False

# Force using residual. This param is to fit tf-official implementation
__C.BACKBONE.MV3.FORCE_RESIDUAL = False

# Sync block act to se module. This param is to fit tf-official implementation
__C.BACKBONE.MV3.SYNC_SE_ACT = True

# Use Conv2dSamePadding to replace Conv2d for fitting tf-original implementation
__C.BACKBONE.MV3.SAME_PAD = False

# ---------------------------------------------------------------------------- #
# PeleeNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.PELEE = AttrDict()

# Channel growth rate
# 24 for peleenet_lite
__C.BACKBONE.PELEE.GROWTH_RATE = 32

# Network initial width
# 24 for peleenet_lite
__C.BACKBONE.PELEE.NUM_INIT = 32

# The number of layers in each block
# (2, 2, 4, 3) for peleenet_lite
__C.BACKBONE.PELEE.LAYERS = (3, 4, 8, 6)

# The initial width of each block
__C.BACKBONE.PELEE.BOTTLENECK_WIDTH = [1, 2, 4, 4]

# Use a squeeze-and-excitation module in each block
__C.BACKBONE.PELEE.USE_SE = False   # TODO

# ---------------------------------------------------------------------------- #
# ProxylessNas options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.PROXY = AttrDict()

# Network setting of ProxylessNas
__C.BACKBONE.PROXY.SETTING = 'mobile'

# Network widen factor
__C.BACKBONE.PROXY.WIDEN_FACTOR = 1.0

# Use a squeeze-and-excitation module in each block
__C.BACKBONE.PROXY.USE_SE = False

# ---------------------------------------------------------------------------- #
# ResNet options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.RESNET = AttrDict()

# The number of layers in each block
# (2, 2, 2, 2) for resnet18 with basicblock
# (3, 4, 6, 3) for resnet34 with basicblock
# (3, 4, 6, 3) for resnet50
# (3, 4, 23, 3) for resnet101
# (3, 8, 36, 3) for resnet152
__C.BACKBONE.RESNET.LAYERS = (3, 4, 6, 3)

# Network initial width
__C.BACKBONE.RESNET.WIDTH = 64

# Network output stride, 32:c5, 16:c4, 8:c3
__C.BACKBONE.RESNET.STRIDE = 32

# Use bottleneck block, False for basicblock
__C.BACKBONE.RESNET.BOTTLENECK = True

# Place the stride 2 conv on the 3x3 filter.
# True for resnet-b
__C.BACKBONE.RESNET.STRIDE_3X3 = False

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnet-c
__C.BACKBONE.RESNET.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnet-d
__C.BACKBONE.RESNET.AVG_DOWN = False

# Use group normalization
__C.BACKBONE.RESNET.USE_GN = False

# Use a aligned module in each block
__C.BACKBONE.RESNET.USE_ALIGN = False

# Type of context module in each block
# 'se' for se, 'gcb' for gcb
__C.BACKBONE.RESNET.STAGE_WITH_CONTEXT = ('none', 'none', 'none', 'none')

# Context module innerplanes ratio
__C.BACKBONE.RESNET.CTX_RATIO = 0.0625

# Type of 3x3 convolution layer in each block
# 'deform' for dcnv1, 'deformv2' for dcnv2
__C.BACKBONE.RESNET.STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')

# ---------------------------------------------------------------------------- #
# ResNeXt options
# ---------------------------------------------------------------------------- #
__C.BACKBONE.RESNEXT = AttrDict()

# The number of layers in each block
# (3, 4, 6, 3) for resnext50
# (3, 4, 23, 3) for resnext101
# (3, 8, 36, 3) for resnext152
__C.BACKBONE.RESNEXT.LAYERS = (3, 4, 6, 3)

# Cardinality (groups) of convolution layers
__C.BACKBONE.RESNEXT.C = 32

# Network initial width of each (conv) group
__C.BACKBONE.RESNEXT.WIDTH = 4

# Use a three (3 * 3) kernels head; False for (7 * 7) kernels head.
# True for resnext-c
__C.BACKBONE.RESNEXT.USE_3x3x3HEAD = False

# Use a (2 * 2) kernels avg_pooling layer in downsampling block.
# True for resnext-d
__C.BACKBONE.RESNEXT.AVG_DOWN = False

# Use group normalization
__C.BACKBONE.RESNEXT.USE_GN = False

# Use a aligned module in each block
__C.BACKBONE.RESNEXT.USE_ALIGN = False

# Type of context module in each block
# 'se' for se, 'gcb' for gcb
__C.BACKBONE.RESNEXT.STAGE_WITH_CONTEXT = ('none', 'none', 'none', 'none')

# Context module innerplanes ratio
__C.BACKBONE.RESNEXT.CTX_RATIO = 0.0625

# Type of 3x3 convolution layer in each block
# 'deform' for dcnv1, 'deformv2' for dcnv2
__C.BACKBONE.RESNEXT.STAGE_WITH_CONV = ('normal', 'normal', 'normal', 'normal')

# ---------------------------------------------------------------------------- #
# VggNet options    # TODO
# ---------------------------------------------------------------------------- #
__C.BACKBONE.VGG = AttrDict()

# Use batch normalization
__C.BACKBONE.VGG.USE_BN = False

# Use group normalization
__C.BACKBONE.VGG.USE_GN = False


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
__C.FPN = AttrDict()

# The Body of FPN to use
# (e.g., "pfpn", "xdeconv")
__C.FPN.BODY = "xdeconv"

# Use C5 or P5 to generate P6
__C.FPN.USE_C5 = True

# Channel dimension of the FPN feature levels
__C.FPN.DIM = 256

# FPN may be used for just RPN, just object detection, or both
# E.g., "conv2"-like level
__C.FPN.LOWEST_BACKBONE_LVL = 2

# E.g., "conv5"-like level
__C.FPN.HIGHEST_BACKBONE_LVL = 5

# Use FPN for RoI transform for object detection if True
__C.FPN.MULTILEVEL_ROIS = True

# Hyperparameters for the RoI-to-FPN level mapping heuristic
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0  # TODO

__C.FPN.ROI_CANONICAL_LEVEL = 4  # k0: where s0 maps to  # TODO

# Coarsest level of the FPN pyramid
__C.FPN.ROI_MAX_LEVEL = 5

# Finest level of the FPN pyramid
__C.FPN.ROI_MIN_LEVEL = 2

# Use FPN for RPN if True
__C.FPN.MULTILEVEL_RPN = False

# Coarsest level of the FPN pyramid
__C.FPN.RPN_MAX_LEVEL = 6

# Finest level of the FPN pyramid
__C.FPN.RPN_MIN_LEVEL = 2

# Use extra FPN levels, as done in the RetinaNet paper
__C.FPN.EXTRA_CONV_LEVELS = False

# Use FPN Lite (dwconv) to replace standard FPN
__C.FPN.USE_LITE = False

# Use BatchNorm in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_BN = False

# Use GroupNorm in the FPN-specific layers (lateral, etc.)
__C.FPN.USE_GN = False

# ---------------------------------------------------------------------------- #
# FPN nasfpn body options
# ---------------------------------------------------------------------------- #
__C.FPN.NASFPN = AttrDict()

# Number of stacking NASFPN module
__C.FPN.NASFPN.NUM_STACK = 7

# Channel dimension of the HRFPN feature levels
__C.FPN.NASFPN.DIM = 256

# Use HRFPN Lite (dwconv) to replace standard HRFPN
__C.FPN.NASFPN.USE_LITE = False

# Use BatchNorm in the HRFPN-specific layers
__C.FPN.NASFPN.USE_BN = False

# Use GroupNorm in the HRFPN-specific layers
__C.FPN.NASFPN.USE_GN = False

# ---------------------------------------------------------------------------- #
# FPN panoptic head options
# ---------------------------------------------------------------------------- #
__C.FPN.PANOPTIC = AttrDict()

# panoptic head conv dim out
__C.FPN.PANOPTIC.CONV_DIM = 256

# Use FPN module before the panoptic head
__C.FPN.PANOPTIC.USE_FPN = True

# Use NASFPN module before the panoptic head
__C.FPN.PANOPTIC.USE_NASFPN = False

# Use BatchNorm in the panoptic head
__C.FPN.PANOPTIC.USE_BN = False

# Use GroupNorm in the panoptic head
__C.FPN.PANOPTIC.USE_GN = False

# ---------------------------------------------------------------------------- #
# FPN deconvx body options
# ---------------------------------------------------------------------------- #
__C.FPN.DECONVX = AttrDict()

# Number of ConvTranspose channels in the deconvx body
__C.FPN.DECONVX.HEAD_DIM = 256

# Decay factor of head ConvTranspose channels
__C.FPN.DECONVX.HEAD_DECAY_FACTOR = 1

# Size of the kernels to use in all ConvTranspose operations
__C.FPN.DECONVX.HEAD_KERNEL = 4

# Number of stacked ConvTranspose layers in deconvx body
__C.FPN.DECONVX.NUM_DECONVS = 3

# Use bias in ConvTranspose layer
__C.FPN.DECONVX.WITH_BIAS = False


# ---------------------------------------------------------------------------- #
# Mask options
# ---------------------------------------------------------------------------- #
__C.MASK = AttrDict()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
__C.MASK.MASK_HEAD = 'simple_none_head'

# Output module of parsing head
__C.MASK.MASK_OUTPUT = 'conv1x1_outputs'

# Output module of parsing loss
__C.MASK.MASK_LOSS = 'mask_loss'

# Mask Number for parsing estimation
__C.MASK.NUM_CLASSES = 81

# Loss weight for mask
__C.MASK.LOSS_WEIGHT = 1

# Use bbox confidence as mask confidence
__C.MASK.USE_BBOX_CONF = False

# ---------------------------------------------------------------------------- #
# Mask gce head options
# ---------------------------------------------------------------------------- #
__C.MASK.GCE_HEAD = AttrDict()

# Hidden Conv layer dimension
__C.MASK.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPPV3
__C.MASK.GCE_HEAD.ASPPV3_DIM = 256

# Dilation for ASPPV3
__C.MASK.GCE_HEAD.ASPPV3_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before ASPPV3
__C.MASK.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3 = 0

# Number of stacked Conv layers in GCE head after ASPPV3
__C.MASK.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3 = 0

# Use NonLocal in the Mask gce head
__C.MASK.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
__C.MASK.GCE_HEAD.NL_RATIO = 1.0

# Use BatchNorm in the Mask gce head
__C.MASK.GCE_HEAD.USE_BN = False

# Use GroupNorm in the Mask gce head
__C.MASK.GCE_HEAD.USE_GN = False


# ---------------------------------------------------------------------------- #
# Keypoint options
# ---------------------------------------------------------------------------- #
__C.KEYPOINT = AttrDict()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
__C.KEYPOINT.KEYPOINT_HEAD = 'simple_none_head'

# Output module of keypoint head
__C.KEYPOINT.KEYPOINT_OUTPUT = 'conv1x1_outputs'

# Use target weight during training
__C.KEYPOINT.USE_TARGET_WEIGHT = True

# Pixel std
__C.KEYPOINT.PIXEL_STD = 200

# Keypoints Number for keypoint estimation
__C.KEYPOINT.NUM_JOINTS = 17

# Soft target type
__C.KEYPOINT.TARGET_TYPE = 'gaussian'

# Usually 1 / 4. size of input
__C.KEYPOINT.HEATMAP_SIZE = (48, 64)

# Sigma
__C.KEYPOINT.SIGMA = 2

# OKS score threshold for testing
__C.KEYPOINT.OKS_THRESH = 0.9

__C.KEYPOINT.IN_VIS_THRESH = 0.2

# Loss weight for keypoint
__C.KEYPOINT.LOSS_WEIGHT = 1

# Use bbox confidence as keypoint confidence
__C.KEYPOINT.USE_BBOX_CONF = False

# ---------------------------------------------------------------------------- #
# Keypoint gce head options
# ---------------------------------------------------------------------------- #
__C.KEYPOINT.GCE_HEAD = AttrDict()

# Hidden Conv layer dimension
__C.KEYPOINT.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPPV3
__C.KEYPOINT.GCE_HEAD.ASPPV3_DIM = 256

# Dilation for ASPPV3
__C.KEYPOINT.GCE_HEAD.ASPPV3_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before ASPPV3
__C.KEYPOINT.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3 = 0

# Number of stacked Conv layers in GCE head after ASPPV3
__C.KEYPOINT.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3 = 0

# Use NonLocal in the Keypoint gce head
__C.KEYPOINT.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
__C.KEYPOINT.GCE_HEAD.NL_RATIO = 1.0

# Use BatchNorm in the Keypoint gce head
__C.KEYPOINT.GCE_HEAD.USE_BN = False

# Use GroupNorm in the Keypoint gce head
__C.KEYPOINT.GCE_HEAD.USE_GN = False


# ---------------------------------------------------------------------------- #
# Parsing options
# ---------------------------------------------------------------------------- #
__C.PARSING = AttrDict()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
__C.PARSING.PARSING_HEAD = 'simple_none_head'

# Output module of parsing head
__C.PARSING.PARSING_OUTPUT = 'conv1x1_outputs'

# Output module of parsing loss
__C.PARSING.PARSING_LOSS = 'parsing_loss'

# Parsing Number for parsing estimation
__C.PARSING.NUM_PARSING = 20

# Minimum score threshold (assuming scores in a [0, 1] range) for semantic
# segmentation results.
# 0.3 for CIHP, 0.05 for MHP-v2
__C.PARSING.SEMSEG_SCORE_THRESH = 0.3

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision parsings
__C.PARSING.SCORE_THRESH = 0.001

# Evaluate the AP metrics
__C.PARSING.EVAL_AP = True

# Index thresh
__C.PARSING.INDEX_THRESH = 0.2

# Loss weight for parsing
__C.PARSING.LOSS_WEIGHT = 1

# Use Parsing IoU for Parsing head
__C.PARSING.PARSINGIOU_ON = False

# Use bbox confidence as parsing confidence
__C.PARSING.USE_BBOX_CONF = False

# ---------------------------------------------------------------------------- #
# Parsing gce head options
# ---------------------------------------------------------------------------- #
__C.PARSING.GCE_HEAD = AttrDict()

# Hidden Conv layer dimension
__C.PARSING.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPPV3
__C.PARSING.GCE_HEAD.ASPPV3_DIM = 256

# Dilation for ASPPV3
__C.PARSING.GCE_HEAD.ASPPV3_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before ASPPV3
__C.PARSING.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3 = 0

# Number of stacked Conv layers in GCE head after ASPPV3
__C.PARSING.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3 = 0

# Use NonLocal in the Keypoint gce head
__C.PARSING.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
__C.PARSING.GCE_HEAD.NL_RATIO = 1.0

# Use BatchNorm in the Keypoint gce head
__C.PARSING.GCE_HEAD.USE_BN = False

# Use GroupNorm in the Keypoint gce head
__C.PARSING.GCE_HEAD.USE_GN = False

# ---------------------------------------------------------------------------- #
# Parsing IoU options
# ---------------------------------------------------------------------------- #
__C.PARSING.PARSINGIOU = AttrDict()

# The head of Parsing IoU to use
# (e.g., "convx_head")
__C.PARSING.PARSINGIOU.PARSINGIOU_HEAD = "convx_head"

# Output module of Parsing IoU head
__C.PARSING.PARSINGIOU.PARSINGIOU_OUTPUT = "linear_output"

# Number of stacked Conv layers in the Parsing IoU head
__C.PARSING.PARSINGIOU.NUM_STACKED_CONVS = 2

# Hidden Conv layer dimension of Parsing IoU head
__C.PARSING.PARSINGIOU.CONV_DIM = 64

# Hidden MLP layer dimension of Parsing IoU head
__C.PARSING.PARSINGIOU.MLP_DIM = 128

# Use BatchNorm in the Parsing IoU head
__C.PARSING.PARSINGIOU.USE_BN = False

# Use GroupNorm in the Parsing IoU head
__C.PARSING.PARSINGIOU.USE_GN = False

# Loss weight for Parsing IoU head
__C.PARSING.PARSINGIOU.LOSS_WEIGHT = 1.0


# ---------------------------------------------------------------------------- #
# UV options
# ---------------------------------------------------------------------------- #
__C.UV = AttrDict()

# The head of model to use
# (e.g., 'simple_none_head', 'gce')
__C.UV.UV_HEAD = 'simple_none_head'

# Output module of UV head
__C.UV.UV_OUTPUT = 'UV_outputs'

# Output module of UV loss
__C.UV.UV_LOSS = 'UV_loss'

# Weights
__C.UV.INDEX_WEIGHTS = 5.0
__C.UV.PART_WEIGHTS = 1.0
__C.UV.POINT_REGRESSION_WEIGHTS = 0.001

# Index thresh
__C.UV.INDEX_THRESH = 0.9

# Use bbox confidence as uv confidence
__C.UV.USE_BBOX_CONF = False

# Evaluate the GPSm metric
__C.UV.GPSM_ON = True

# ---------------------------------------------------------------------------- #
# UV gce head options
# ---------------------------------------------------------------------------- #
__C.UV.GCE_HEAD = AttrDict()

# Hidden Conv layer dimension
__C.UV.GCE_HEAD.CONV_DIM = 512

# Dimension for ASPPV3
__C.UV.GCE_HEAD.ASPPV3_DIM = 256

# Dilation for ASPPV3
__C.UV.GCE_HEAD.ASPPV3_DILATION = (6, 12, 18)

# Number of stacked Conv layers in GCE head before ASPPV3
__C.UV.GCE_HEAD.NUM_CONVS_BEFORE_ASPPV3 = 0

# Number of stacked Conv layers in GCE head after ASPPV3
__C.UV.GCE_HEAD.NUM_CONVS_AFTER_ASPPV3 = 0

# Use NonLocal in the UV gce head
__C.UV.GCE_HEAD.USE_NL = False

# Reduction ration of nonlocal
__C.UV.GCE_HEAD.NL_RATIO = 1.0

# Use BatchNorm in the UV gce head
__C.UV.GCE_HEAD.USE_BN = False

# Use GroupNorm in the UV gce head
__C.UV.GCE_HEAD.USE_GN = False


# ---------------------------------------------------------------------------- #
# Visualization options
# ---------------------------------------------------------------------------- #
__C.VIS = AttrDict()

# Dump detection visualizations
__C.VIS.ENABLED = False

# Score threshold for visualization
__C.VIS.VIS_TH = 0.9

# ---------------------------------------------------------------------------- #
# Show box options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_BOX = AttrDict()

# Visualizing detection bboxes
__C.VIS.SHOW_BOX.ENABLED = True

# Visualization color scheme
# 'green', 'category' or 'instance'
__C.VIS.SHOW_BOX.COLOR_SCHEME = 'green'

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
__C.VIS.SHOW_BOX.COLORMAP = 'COCO81'

# Border thick
__C.VIS.SHOW_BOX.BORDER_THICK = 2

# ---------------------------------------------------------------------------- #
# Show class options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_CLASS = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_CLASS.ENABLED = True

# Default: gray
__C.VIS.SHOW_CLASS.COLOR = (218, 227, 218)

# Font scale of class string
__C.VIS.SHOW_CLASS.FONT_SCALE = 0.45

# ---------------------------------------------------------------------------- #
# Show Mask options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_MASK = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_MASK.ENABLED = True

# False = (255, 255, 255) = white
__C.VIS.SHOW_MASK.MASK_COLOR_FOLLOW_BOX = True

# Mask ahpha
__C.VIS.SHOW_MASK.MASK_ALPHA = 0.4

# Whether show border
__C.VIS.SHOW_MASK.SHOW_BORDER = True

# Border color, (255, 255, 255) for white, (0, 0, 0) for black
__C.VIS.SHOW_MASK.BORDER_COLOR = (255, 255, 255)

# Border thick
__C.VIS.SHOW_MASK.BORDER_THICK = 2

# ---------------------------------------------------------------------------- #
# Show keypoints options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_KPS = AttrDict()

# Visualizing detection keypoints
__C.VIS.SHOW_KPS.ENABLED = True

# Keypoints threshold
__C.VIS.SHOW_KPS.KPS_TH = 0.4

# Default: white
__C.VIS.SHOW_KPS.KPS_COLOR_WITH_PARSING = (255, 255, 255)

# Keypoints alpha
__C.VIS.SHOW_KPS.KPS_ALPHA = 0.7

# Link thick
__C.VIS.SHOW_KPS.LINK_THICK = 2

# Circle radius
__C.VIS.SHOW_KPS.CIRCLE_RADIUS = 3

# Circle thick
__C.VIS.SHOW_KPS.CIRCLE_THICK = -1

# ---------------------------------------------------------------------------- #
# Show parsing options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_PARSS = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_PARSS.ENABLED = True

# Color map, 'COCO81', 'VOC21', 'ADE151', 'LIP20', 'MHP59'
__C.VIS.SHOW_PARSS.COLORMAP = 'CIHP20'

# Parsing alpha
__C.VIS.SHOW_PARSS.PARSING_ALPHA = 0.4

# Whether show border
__C.VIS.SHOW_PARSS.SHOW_BORDER = True

# Border color
__C.VIS.SHOW_PARSS.BORDER_COLOR = (255, 255, 255)

# Border thick
__C.VIS.SHOW_PARSS.BORDER_THICK = 1

# ---------------------------------------------------------------------------- #
# Show uv options
# ---------------------------------------------------------------------------- #
__C.VIS.SHOW_UV = AttrDict()

# Visualizing detection classes
__C.VIS.SHOW_UV.ENABLED = True

# Whether show border
__C.VIS.SHOW_UV.SHOW_BORDER = True

# Border thick
__C.VIS.SHOW_UV.BORDER_THICK = 6

# Grid thick
__C.VIS.SHOW_UV.GRID_THICK = 2

# Grid lines num
__C.VIS.SHOW_UV.LINES_NUM = 15


# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the full config key as a string to the set below.
# ---------------------------------------------------------------------------- #
_DEPCRECATED_KEYS = set()


# ---------------------------------------------------------------------------- #
# Renamed options
# If you rename a config option, record the mapping from the old name to the new
# name in the dictionary below. Optionally, if the type also changed, you can
# make the value a tuple that specifies first the renamed key and then
# instructions for how to edit the config file.
# ---------------------------------------------------------------------------- #
_RENAMED_KEYS = {
    'EXAMPLE.RENAMED.KEY': 'EXAMPLE.KEY',  # Dummy example to follow
    'PIXEL_MEAN': 'PIXEL_MEANS',
    'PIXEL_STD': 'PIXEL_STDS',
}


def merge_cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        if _key_is_deprecated(full_key):
            continue
        if _key_is_renamed(full_key):
            _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            if _key_is_deprecated(full_key):
                continue
            elif _key_is_renamed(full_key):
                _raise_key_rename_error(full_key)
            else:
                raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a


def _key_is_deprecated(full_key):
    if full_key in _DEPCRECATED_KEYS:
        return True
    return False


def _key_is_renamed(full_key):
    return full_key in _RENAMED_KEYS


def _raise_key_rename_error(full_key):
    new_key = _RENAMED_KEYS[full_key]
    if isinstance(new_key, tuple):
        msg = ' Note: ' + new_key[1]
        new_key = new_key[0]
    else:
        msg = ''
    raise KeyError(
        'Key {} was renamed to {}; please update your config.{}'.
            format(full_key, new_key, msg)
    )
