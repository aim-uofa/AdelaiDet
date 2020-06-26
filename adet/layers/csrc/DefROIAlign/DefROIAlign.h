#pragma once
#include <torch/types.h>

namespace adet {

#ifdef WITH_CUDA
at::Tensor DefROIAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const at::Tensor& offsets,  // def added
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float trans_std,  // def added
    bool aligned);

at::Tensor DefROIAlign_backward_cuda(
    const at::Tensor& input,  // def added
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& offsets,  // def added
    const at::Tensor& grad_offsets,  // def added
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    const float trans_std,  // def added
    bool aligned);
#endif

// Interface for Python
inline at::Tensor DefROIAlign_forward(
    const at::Tensor& input,
    const at::Tensor& rois,
    const at::Tensor& offsets,  // def added
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const float trans_std,  // def added
    bool aligned) {
  if (input.type().is_cuda()) {
#ifdef WITH_CUDA
    return DefROIAlign_forward_cuda(
        input,
        rois,
        offsets,
        spatial_scale,
        pooled_height,
        pooled_width,
        sampling_ratio,
        trans_std,
        aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not supported");
}

inline at::Tensor DefROIAlign_backward(
    const at::Tensor& input,  // def added
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& offsets,  // def added
    const at::Tensor& grad_offsets,  // def added
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    const float trans_std,  // def added
    bool aligned) {
  if (grad.type().is_cuda()) {
#ifdef WITH_CUDA
    return DefROIAlign_backward_cuda(
        input,  // def added
        grad,
        rois,
        offsets,  // def added
        grad_offsets, // def added
        spatial_scale,
        pooled_height,
        pooled_width,
        batch_size,
        channels,
        height,
        width,
        sampling_ratio,
        trans_std, // def added
        aligned);
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }
  AT_ERROR("CPU version not supported");
}

} // namespace adet
