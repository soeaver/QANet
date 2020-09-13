#include "soft_nms.h"

namespace {

template <typename T>
void ml_soft_nms_cpu_kernel(
    int& ndets,
    T* dets,
    T* areas,
    T* scores,
    int64_t* labels,
    int64_t* inds,
    const float threshold,
    const int method,
    const float sigma,
    const float min_score,
    const int topk) {
  T* x1 = dets + 0;
  T* y1 = dets + 1;
  T* x2 = dets + 2;
  T* y2 = dets + 3;

  for (int i = 0, i_box = 0; i < ndets; i++, i_box += 4) {
    areas[i] = (x2[i_box] - x1[i_box]) * (y2[i_box] - y1[i_box]);
  }

  int64_t iind;
  int pos, pos_box, max_pos, max_pos_box, ndets_box = ndets * 4;
  T ix1, iy1, ix2, iy2, iscore, ilabel, iarea, max_score, inter, ovr;
  for (int i = 0, i_box = 0; i < ndets; i++, i_box += 4) {
    // keep topk
    if (topk == i) {
      ndets = topk;
      break;
    }

    // get max box
    max_pos = i;
    max_score = scores[i];
    for (pos = i + 1; pos < ndets; pos++) {
      if (max_score < scores[pos]) {
        max_score = scores[pos];
        max_pos = pos;
      }
    }
    max_pos_box = max_pos * 4;

    // add max box as a detection
    ix1 = x1[max_pos_box];
    iy1 = y1[max_pos_box];
    ix2 = x2[max_pos_box];
    iy2 = y2[max_pos_box];
    iscore = scores[max_pos];
    ilabel = labels[max_pos];
    iarea = areas[max_pos];
    iind = inds[max_pos];

    // swap ith box with position of max box
    x1[max_pos_box] = x1[i_box];
    y1[max_pos_box] = y1[i_box];
    x2[max_pos_box] = x2[i_box];
    y2[max_pos_box] = y2[i_box];
    scores[max_pos] = scores[i];
    labels[max_pos] = labels[i];
    areas[max_pos] = areas[i];
    inds[max_pos] = inds[i];

    x1[i_box] = ix1;
    y1[i_box] = iy1;
    x2[i_box] = ix2;
    y2[i_box] = iy2;
    scores[i] = iscore;
    labels[i] = ilabel;
    areas[i] = iarea;
    inds[i] = iind;

    // NMS iterations, note that N changes
    // if detection boxes fall below threshold
    for (pos = i + 1, pos_box = i_box + 4; pos < ndets; pos++, pos_box += 4) {
      if (ilabel == labels[pos]) {
        inter =
          std::max((T)0, std::min(ix2, x2[pos_box]) - std::max(ix1, x1[pos_box])) *
          std::max((T)0, std::min(iy2, y2[pos_box]) - std::max(iy1, y1[pos_box]));
        ovr = inter / (iarea + areas[pos] - inter);

        switch (method) {
            case 1:  // linear
            if (ovr > threshold) {
              scores[pos] = ((T)1 - ovr) * scores[pos];
            }
            break;
            case 2:  // gaussian
            scores[pos] =
              std::exp(-(ovr * ovr) / sigma) * scores[pos];
            break;
            default:  // hard and other methods
            if (ovr > threshold) {
              scores[pos] = (T)0;
            }
        }
      }

      // if box score falls below threshold, discard the box by
      // swapping with last box update N
      if (scores[pos] < min_score) {
        ndets--;
        ndets_box -= 4;
        x1[pos_box] = x1[ndets_box];
        y1[pos_box] = y1[ndets_box];
        x2[pos_box] = x2[ndets_box];
        y2[pos_box] = y2[ndets_box];
        scores[pos] = scores[ndets];
        labels[pos] = labels[ndets];
        areas[pos] = areas[ndets];
        inds[pos] = inds[ndets];
        pos--;
        pos_box -= 4;
      }
    }
  }
}

} // namespace

namespace pet {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ml_soft_nms_cpu(
    const at::Tensor& dets,
    const at::Tensor& scores,
    const at::Tensor& labels,
    const float threshold,
    const int method,
    const float sigma,
    const float min_score,
    const int topk) {
  AT_ASSERTM(!dets.is_cuda(), "dets must be a CPU tensor");
  AT_ASSERTM(!scores.is_cuda(), "scores must be a CPU tensor");
  AT_ASSERTM(!labels.is_cuda(), "labels must be a CPU tensor");
  AT_ASSERTM(
      dets.scalar_type() == scores.scalar_type(),
      "dets should have the same type as scores");

  if (dets.numel() == 0) {
      return std::make_tuple(
          at::empty({0, 4}, dets.options()),
          at::empty({0}, dets.options()),
          at::empty({0}, dets.options().dtype(at::kLong)),
          at::empty({0}, dets.options().dtype(at::kLong)));
  }

  int ndets = dets.size(0);

  auto dets_t = dets.contiguous();
  auto scores_t = scores.contiguous();
  auto labels_t = labels.contiguous();

  at::Tensor areas_t = at::empty(ndets, dets.options());
  at::Tensor inds_t = at::arange(ndets, dets.options().dtype(at::kLong));

  AT_DISPATCH_FLOATING_TYPES(dets.scalar_type(), "ml_soft_nms", [&] {
    ml_soft_nms_cpu_kernel<scalar_t>(
        ndets,
        dets_t.data_ptr<scalar_t>(),
        areas_t.data_ptr<scalar_t>(),
        scores_t.data_ptr<scalar_t>(),
        labels_t.data_ptr<int64_t>(),
        inds_t.data_ptr<int64_t>(),
        threshold,
        method,
        sigma,
        min_score,
        topk);
  });

  at::Tensor res_dets = at::zeros({ndets, 4}, dets.options());
  at::Tensor res_scores = at::zeros({ndets}, dets.options());
  at::Tensor res_labels = at::zeros({ndets}, dets.options().dtype(at::kLong));
  at::Tensor res_inds = at::zeros({ndets}, dets.options().dtype(at::kLong));

  res_dets = dets_t.slice(0, 0, ndets);
  res_scores = scores_t.slice(0, 0, ndets);
  res_labels = labels_t.slice(0, 0, ndets);
  res_inds = inds_t.slice(0, 0, ndets);

  return std::make_tuple(res_dets, res_scores, res_labels, res_inds);
}

} // namespace pet
