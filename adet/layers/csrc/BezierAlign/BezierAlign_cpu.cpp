// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#include <ATen/TensorUtils.h>
#include "BezierAlign.h"

namespace {

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
  int pos1;
  int pos2;
  int pos3;
  int pos4;
  T w1;
  T w2;
  T w3;
  T w4;
};


template <typename T>
T bezier_curve(
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
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T p0_x, T p0_y, T p1_x, T p1_y,
    T p2_x, T p2_y, T p3_x, T p3_y,
    T p4_x, T p4_y, T p5_x, T p5_y,
    T p6_x, T p6_y, T p7_x, T p7_y,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      // compute the coords
      const T u = pw / static_cast<T>(pooled_width);
      const T v = ph / static_cast<T>(pooled_height);
      const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
      const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
      const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
      const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
      const T x_center = x1 * v + x0 * (1. - v);
      const T y_center = y1 * v + y0 * (1. - v);
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = y_center - (T)0.5 * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h /
                static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = x_center - (T)0.5 * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w /
                  static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.pos1 = 0;
            pc.pos2 = 0;
            pc.pos3 = 0;
            pc.pos4 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

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
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc<T> pc;
          pc.pos1 = y_low * width + x_low;
          pc.pos2 = y_low * width + x_high;
          pc.pos3 = y_high * width + x_low;
          pc.pos4 = y_high * width + x_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;

          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void BezierAlignForward(
    const int nthreads,
    const T* input,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* rois,
    T* output,
    bool aligned) {
  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    // beziers have size Nx(1+8*2) = Nx17
    const T* offset_rois = rois + n * 17;
    int roi_batch_ind = offset_rois[0];

    T offset = aligned ? (T)0.5 : (T)0.0;
    // Do not use rounding; this implementation detail is critical
    T p0_x = offset_rois[1] * spatial_scale;
    T p0_y = offset_rois[2] * spatial_scale;
    T p1_x = offset_rois[3] * spatial_scale;
    T p1_y = offset_rois[4] * spatial_scale;
    T p2_x = offset_rois[5] * spatial_scale;
    T p2_y = offset_rois[6] * spatial_scale;
    T p3_x = offset_rois[7] * spatial_scale;
    T p3_y = offset_rois[8] * spatial_scale;
    T p4_x = offset_rois[15] * spatial_scale;
    T p4_y = offset_rois[16] * spatial_scale;
    T p5_x = offset_rois[13] * spatial_scale;
    T p5_y = offset_rois[14] * spatial_scale;
    T p6_x = offset_rois[11] * spatial_scale;
    T p6_y = offset_rois[12] * spatial_scale;
    T p7_x = offset_rois[9 ] * spatial_scale;
    T p7_y = offset_rois[10] * spatial_scale;

    T roi_width = std::max(std::abs(p0_x - p3_x), std::abs(p4_x - p7_x));
    T roi_height = std::max(std::abs(p0_y - p3_y), std::abs(p4_y - p7_y));
    if (aligned) {
      AT_ASSERTM(
          roi_width >= 0 && roi_height >= 0,
          "ROIs in ROIAlign cannot have non-negative size!");
    } else { // for backward-compatibility only
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros == 0/1, instead of NaN.
    const T count = std::max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        p0_x, p0_y, p1_x, p1_y,
        p2_x, p2_y, p3_x, p3_y,
        p4_x, p4_y, p5_x, p5_y,
        p6_x, p6_y, p7_x, p7_y,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);

    for (int c = 0; c < channels; c++) {
      int index_n_c = index_n + c * pooled_width * pooled_height;
      const T* offset_input =
          input + (roi_batch_ind * channels + c) * height * width;
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          int index = index_n_c + ph * pooled_width + pw;

          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];
              output_val += pc.w1 * offset_input[pc.pos1] +
                  pc.w2 * offset_input[pc.pos2] +
                  pc.w3 * offset_input[pc.pos3] + pc.w4 * offset_input[pc.pos4];

              pre_calc_index += 1;
            }
          }
          output_val /= count;

          output[index] = output_val;
        } // for pw
      } // for ph
    } // for c
  } // for n
}

template <typename T>
void bilinear_interpolate_gradient(
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
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}

template <class T>
inline void add(T* address, const T& val) {
  *address += val;
}

template <typename T>
void BezierAlignBackward(
    const int nthreads,
    const T* grad_output,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    T* grad_input,
    const T* rois,
    const int n_stride,
    const int c_stride,
    const int h_stride,
    const int w_stride,
    bool aligned) {
  for (int index = 0; index < nthreads; index++) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 17;
    int roi_batch_ind = offset_rois[0];

    // Do not use rounding; this implementation detail is critical
    T offset = aligned ? (T)0.5 : (T)0.0;
    T p0_x = offset_rois[1] * spatial_scale;
    T p0_y = offset_rois[2] * spatial_scale;
    T p1_x = offset_rois[3] * spatial_scale;
    T p1_y = offset_rois[4] * spatial_scale;
    T p2_x = offset_rois[5] * spatial_scale;
    T p2_y = offset_rois[6] * spatial_scale;
    T p3_x = offset_rois[7] * spatial_scale;
    T p3_y = offset_rois[8] * spatial_scale;
    T p4_x = offset_rois[15] * spatial_scale;
    T p4_y = offset_rois[16] * spatial_scale;
    T p5_x = offset_rois[13] * spatial_scale;
    T p5_y = offset_rois[14] * spatial_scale;
    T p6_x = offset_rois[11] * spatial_scale;
    T p6_y = offset_rois[12] * spatial_scale;
    T p7_x = offset_rois[9 ] * spatial_scale;
    T p7_y = offset_rois[10] * spatial_scale;

    // compute the coords
    const T u = pw / static_cast<T>(pooled_width);
    const T v = ph / static_cast<T>(pooled_height);
    const T x0 = bezier_curve(p0_x, p1_x, p2_x, p3_x, u);
    const T y0 = bezier_curve(p0_y, p1_y, p2_y, p3_y, u);
    const T x1 = bezier_curve(p4_x, p5_x, p6_x, p7_x, u);
    const T y1 = bezier_curve(p4_y, p5_y, p6_y, p7_y, u);
    const T x_center = x1 * v + x0 * (1. - v) - offset;
    const T y_center = y1 * v + y0 * (1. - v) - offset;

    T roi_width = std::max(std::abs(p0_x - p3_x), std::abs(p4_x - p7_x));
    T roi_height = std::max(std::abs(p0_y - p3_y), std::abs(p4_y - p7_y));
    if (aligned) {
      AT_ASSERTM(
          roi_width >= 0 && roi_height >= 0,
          "ROIs in ROIAlign do not have non-negative size!");
    } else { // for backward-compatibility only
      roi_width = std::max(roi_width, (T)1.);
      roi_height = std::max(roi_height, (T)1.);
    }
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    T* offset_grad_input =
        grad_input + ((roi_batch_ind * channels + c) * height * width);

    int output_offset = n * n_stride + c * c_stride;
    const T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy++) {
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

        T g1 = grad_output_this_bin * w1 / count;
        T g2 = grad_output_this_bin * w2 / count;
        T g3 = grad_output_this_bin * w3 / count;
        T g4 = grad_output_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          // atomic add is not needed for now since it is single threaded
          add(offset_grad_input + y_low * width + x_low, static_cast<T>(g1));
          add(offset_grad_input + y_low * width + x_high, static_cast<T>(g2));
          add(offset_grad_input + y_high * width + x_low, static_cast<T>(g3));
          add(offset_grad_input + y_high * width + x_high, static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  } // for
} // BezierAlignBackward

} // namespace

namespace adet {

at::Tensor BezierAlign_forward_cpu(
    const at::Tensor& input,
    const at::Tensor& rois,
    const float spatial_scale,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    bool aligned) {
  AT_ASSERTM(input.device().is_cpu(), "input must be a CPU tensor");
  AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "BezierAlign_forward_cpu";
  at::checkAllSameType(c, {input_t, rois_t});

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());

  auto output_size = num_rois * pooled_height * pooled_width * channels;

  if (output.numel() == 0)
    return output;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.type(), "BezierAlign_forward", [&] {
    BezierAlignForward<scalar_t>(
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
  return output;
}

at::Tensor BezierAlign_backward_cpu(
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
  AT_ASSERTM(grad.device().is_cpu(), "grad must be a CPU tensor");
  AT_ASSERTM(rois.device().is_cpu(), "rois must be a CPU tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "BezierAlign_backward_cpu";
  at::checkAllSameType(c, {grad_t, rois_t});

  at::Tensor grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    return grad_input;
  }

  // get stride values to ensure indexing into gradients is correct.
  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad.type(), "BezierAlign_forward", [&] {
    BezierAlignBackward<scalar_t>(
        grad.numel(),
        grad.contiguous().data_ptr<scalar_t>(),
        spatial_scale,
        channels,
        height,
        width,
        pooled_height,
        pooled_width,
        sampling_ratio,
        grad_input.data_ptr<scalar_t>(),
        rois.contiguous().data_ptr<scalar_t>(),
        n_stride,
        c_stride,
        h_stride,
        w_stride,
        aligned);
  });
  return grad_input;
}

} // namespace adet
