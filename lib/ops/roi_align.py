import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.modules.utils import _pair

from lib.ops import _C

INTERPOLATION_METHOD = {"bilinear": 0, "nearest": 1}


class _ROIAlign(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned, interpolation="bilinear"):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        ctx.interpolation_method = INTERPOLATION_METHOD[interpolation]
        output = _C.roi_align_forward(
            input,
            roi,
            spatial_scale,
            output_size[0],
            output_size[1],
            sampling_ratio,
            aligned,
            ctx.interpolation_method,
        )
        return output

    @staticmethod
    @custom_bwd
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        aligned = ctx.aligned
        interpolation_method = ctx.interpolation_method
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward(
            grad_output,
            rois,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
            aligned,
            interpolation_method,
        )
        return grad_input, None, None, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned, interpolation="bilinear"):
        assert interpolation in INTERPOLATION_METHOD, "Unknown interpolation method: {}".format(interpolation)
        super(ROIAlign, self).__init__()
        self.output_size = _pair(output_size)
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned
        self.interpolation_method = interpolation

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return roi_align(
            input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned, self.interpolation_method
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
