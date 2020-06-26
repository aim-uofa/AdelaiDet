import torch
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from adet import _C


class _DefROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, offsets, output_size, spatial_scale, sampling_ratio, trans_std, aligned):
        ctx.save_for_backward(input, roi, offsets)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.trans_std = trans_std
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        output = _C.def_roi_align_forward(
            input, roi, offsets, spatial_scale, output_size[0], output_size[1],
            sampling_ratio, trans_std, aligned
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        data, rois, offsets = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        trans_std = ctx.trans_std
        bs, ch, h, w = ctx.input_shape
        grad_offsets = torch.zeros_like(offsets)

        grad_input = _C.def_roi_align_backward(
            data,
            grad_output,
            rois,
            offsets,
            grad_offsets,
            spatial_scale,
            output_size[0],
            output_size[1],
            bs,
            ch,
            h,
            w,
            sampling_ratio,
            trans_std,
            ctx.aligned,
        )
        return grad_input, None, grad_offsets, None, None, None, None, None


def_roi_align = _DefROIAlign.apply


class DefROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale,
                 sampling_ratio, trans_std, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                sample. 0 to take samples densely.
            trans_std (float): offset scale according to the normalized roi size
            aligned (bool): if False, use the legacy implementation in
                Detectron. If True, align the results more perfectly.
        """
        super(DefROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.trans_std = trans_std
        self.aligned = aligned

    def forward(self, input, rois, offsets):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return def_roi_align(
            input, rois, offsets, self.output_size,
            self.spatial_scale, self.sampling_ratio,
            self.trans_std, self.aligned
        )

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", trans_std=" + str(self.trans_std)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
