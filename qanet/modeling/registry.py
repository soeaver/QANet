from lib.utils.registry import Registry


"""
Feature Extractor.
"""
# Backbone
BACKBONES = Registry()

# FPN
FPN_BODY = Registry()


"""
Instance Head.
"""
# Mask Head
MASK_HEADS = Registry()
MASK_OUTPUTS = Registry()
MASKIOU_HEADS = Registry()
MASKIOU_OUTPUTS = Registry()

# Keypoint Head
KEYPOINT_HEADS = Registry()
KEYPOINT_OUTPUTS = Registry()

# Parsing Head
PARSING_HEADS = Registry()
PARSING_OUTPUTS = Registry()
PARSINGIOU_HEADS = Registry()
PARSINGIOU_OUTPUTS = Registry()

# UV Head
UV_HEADS = Registry()
UV_OUTPUTS = Registry()
