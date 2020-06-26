// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAApplyUtils.cuh>

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ T bezier_curve(
    const T p0,
    const T p1,
    const T p2,
    const T p3,
    const T u) {
  return (
      (1. - u) * (1. - u) * (1. - u) * p0
    + 3. * u * (1. - u) * (1. - u) * p1
    + 3. * u * u * (1. - u) * p2
    + u * u * u * p3);
}

template <typename T>
__device__ T bilinear_interpolate(
    const T* bottom_data,
    const int height,
    const int width,
    T y,
    T x,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T>
__global__ void BezierAlignForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,  // bottom rois contains the bezier curve
    T* top_data,
    bool aligned) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // beziers have size Nx(1+8*2) = Nx17
    const T* offset_bottom_rois = bottom_rois + n * 17;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;

    // TODO: avoid this by using parallel annotation, for good
    T p0_x = offset_bottom_rois[1 ] * spatial_scale;
    T p0_y = offset_bottom_rois[2 ] * spatial_scale;
    T p1_x = offset_bottom_rois[3 ] * spatial_scale;
    T p1_y = offset_bottom_rois[4 ] * spatial_scale;
    T p2_x = offset_bottom_rois[5 ] * spatial_scale;
    T p2_y = offset_bottom_rois[6 ] * spatial_scale;
    T p3_x = offset_bottom_rois[7 ] * spatial_scale;
    T p3_y = offset_bottom_rois[8 ] * spatial_scale;
    T p4_x = offset_bottom_rois[15] * spatial_scale;
    T p4_y = offset_bottom_rois[16] * spatial_scale;
    T p5_x = offset_bottom_rois[13] * spatial_scale;
    T p5_y = offset_bottom_rois[14] * spatial_scale;
    T p6_x = offset_bottom_rois[11] * spatial_scale;
    T p6_y = offset_bottom_rois[12] * spatial_scale;
    T p7_x = offset_bottom_rois[9 ] * spatial_scale;
    T p7_y = offset_bottom_rois[10] * spatial_scale;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
    const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
    const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
    const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
    const T x_center = x1 * v + x0 * (1. - v) - offset;
    const T y_center = y1 * v + y0 * (1. - v) - offset;
    
    T roi_width = max(abs(p0_x - p3_x), abs(p4_x - p7_x));
    T roi_height = max(abs(p0_y - p3_y), abs(p4_y - p7_y));
    if (!aligned) { // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    const T* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros == 0/1, instead of NaN.
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = y_center - (T)0.5 * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = x_center - (T)0.5 * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(
            offset_bottom_data, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    top_data[index] = output_val;
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height,
    const int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    const int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <typename T>
__global__ void BezierAlignBackwardFeature(
    const int nthreads,
    const T* top_diff,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois,
    bool aligned) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    // beziers have size Nx(1+8*2) = Nx17
    const T* offset_bottom_rois = bottom_rois + n * 17;
    int roi_batch_ind = offset_bottom_rois[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T p0_x = offset_bottom_rois[1 ] * spatial_scale;
    T p0_y = offset_bottom_rois[2 ] * spatial_scale;
    T p1_x = offset_bottom_rois[3 ] * spatial_scale;
    T p1_y = offset_bottom_rois[4 ] * spatial_scale;
    T p2_x = offset_bottom_rois[5 ] * spatial_scale;
    T p2_y = offset_bottom_rois[6 ] * spatial_scale;
    T p3_x = offset_bottom_rois[7 ] * spatial_scale;
    T p3_y = offset_bottom_rois[8 ] * spatial_scale;
    T p4_x = offset_bottom_rois[15] * spatial_scale;
    T p4_y = offset_bottom_rois[16] * spatial_scale;
    T p5_x = offset_bottom_rois[13] * spatial_scale;
    T p5_y = offset_bottom_rois[14] * spatial_scale;
    T p6_x = offset_bottom_rois[11] * spatial_scale;
    T p6_y = offset_bottom_rois[12] * spatial_scale;
    T p7_x = offset_bottom_rois[9 ] * spatial_scale;
    T p7_y = offset_bottom_rois[10] * spatial_scale;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
    const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
    const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
    const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
    const T x_center = x1 * v + x0 * (1. - v) - offset;
    const T y_center = y1 * v + y0 * (1. - v) - offset;

    T roi_width = max(abs(p0_x - p3_x), abs(p4_x - p7_x));
    T roi_height = max(abs(p0_y - p3_y), abs(p4_y - p7_y));
    if (!aligned) { // for backward-compatibility only
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_bottom_diff =
        bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = y_center - (T)0.5 * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = x_center - (T)0.5 * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          atomicAdd(
              offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
          atomicAdd(
              offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
          atomicAdd(
              offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward

namespace adet {

at::Tensor BezierAlign_forward_cuda(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    bool aligned) {
  AT_ASSERTM(input.device().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.device().is_cuda(), "rois must be a CUDA tensor");
  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ROIAlign_forward_cuda";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});
  at::cuda::CUDAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "BezierAlign_forward", [&] {
    BezierAlignForward<scalar_t><<<grid, block, 0, stream>>>(
        output_size,
        input.contiguous().data_ptr<scalar_t>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        rois.contiguous().data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        aligned);
  });
  cudaDeviceSynchronize();
  AT_CUDA_CHECK(cudaGetLastError());
  return output;
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor BezierAlign_backward_cuda(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int batch_size,
    const int channels,
    const int height,
    const int width,
    const int sampling_ratio,
    bool aligned) {
  AT_ASSERTM(grad.device().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.device().is_cuda(), "rois must be a CUDA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};
  at::CheckedFrom c = "ROIAlign_backward_cuda";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});
  at::cuda::CUDAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);
  auto grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(
      at::cuda::ATenCeilDiv(
          static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_CUDA_CHECK(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "BezierAlign_backward", [&] {
    BezierAlignBackwardFeature<scalar_t><<<grid, block, 0, stream>>>(
        grad.numel(),
        grad.contiguous().data_ptr<scalar_t>(),
        num_rois,
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        grad_input.data_ptr<scalar_t>(),
        rois.contiguous().data_ptr<scalar_t>(),
        aligned);
  });
  AT_CUDA_CHECK(cudaGetLastError());
  return grad_input;
}

} // namespace detectron2
