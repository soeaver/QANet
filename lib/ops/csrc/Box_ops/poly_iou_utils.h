#pragma once

#include <iostream>
// #include <vector>

#ifdef __CUDACC__
// Designates functions callable from the host (CPU) and the device (GPU)
#define HOST_DEVICE __host__ __device__
#define HOST_DEVICE_INLINE HOST_DEVICE __forceinline__
#else
#include <algorithm>
#define HOST_DEVICE
#define HOST_DEVICE_INLINE HOST_DEVICE inline
#endif

namespace pet {

namespace {

#define MAXN 10
float const eps = 1e-8;

template <typename T>
HOST_DEVICE_INLINE int
sig(T d) {
  return (d > eps) - (d < -eps);
}

template <typename T>
HOST_DEVICE_INLINE bool
point_eq(const float2 a, const float2 b) {
  return sig<T>(a.x - b.x) == 0 && sig<T>(a.y - b.y) == 0;
}

template <typename T>
HOST_DEVICE_INLINE T
cross(float2 o, float2 a, float2 b) {
  return (a.x - o.x) * (b.y - o.y) - (b.x - o.x) * (a.y - o.y);
}

template <typename T>
HOST_DEVICE_INLINE T
area(float2* ps, int n) {
  ps[n] = ps[0];
  T res = 0;
  for(int i = 0; i < n; i++) {
    res += ps[i].x * ps[i + 1].y - ps[i].y * ps[i + 1].x;
  }
  return res * 0.5;
}

template <typename T>
HOST_DEVICE_INLINE void
polygon_cut(float2* p, int& n, float2 a, float2 b, float2* pp) {
  int m = 0;
  p[n] = p[0];
  for (int i = 0; i < n; i++) {
    float s1, s2, s3;
    s1 = cross<T>(a, b, p[i]);
    s2 = cross<T>(a, b, p[i + 1]);
    s3 = s2 - s1;
    if (sig<T>(s1) > 0) {
      pp[m++] = p[i];
    }
    if (sig<T>(s1) != sig<T>(s2) && sig<T>(s3) != 0) {
      pp[m].x = (p[i].x * s2 - p[i + 1].x * s1) / s3;
      pp[m].y = (p[i].y * s2 - p[i + 1].y * s1) / s3;
      m++;
    }
  }

  n = 0;
  for (int i = 0; i < m; i++) {
    if (!i || !(point_eq<T>(pp[i], pp[i - 1]))) {
      p[n++] = pp[i];
    }
  }

  while (n > 1 && point_eq<T>(p[n-1], p[0])) {
    n--;
  }
}

template <typename T>
HOST_DEVICE_INLINE void
point_swap(float2* a, float2* b) {
  float2 temp = *a;
  *a = *b;
  *b = temp;
}

template <typename T>
HOST_DEVICE_INLINE void
point_reverse(float2* first, float2* last) {
  while ((first != last) && (first != --last)) {
      point_swap<T>(first, last);
      ++first;
  }
}

template <typename T>
HOST_DEVICE_INLINE T
intersect_area(float2 a, float2 b, float2 c, float2 d) {
  float2 o = make_float2(0, 0);
  int s1 = sig<T>(cross<T>(o, a, b));
  int s2 = sig<T>(cross<T>(o, c, d));

  if(s1 == 0 || s2 == 0) {
    return 0.0;
  }

  if (s1 == -1) point_swap<T>(&a, &b);
  if (s2 == -1) point_swap<T>(&c, &d);

  float2 p[10] = {o, a, b};
  int n = 3;
  float2 pp[MAXN];
  polygon_cut<T>(p, n, o, c, pp);
  polygon_cut<T>(p, n, c, d, pp);
  polygon_cut<T>(p, n, d, o, pp);

  float res = fabs(area<T>(p, n));
  if (s1 * s2 == -1) {
    res = -res;
  }
  return res;
}

template <typename T>
HOST_DEVICE_INLINE T
poly_intersection(float2* ps1, float2* ps2) {
  if (area<T>(ps1, 4) < 0) {
    point_reverse<T>(ps1, ps1 + 4);
  }
  if (area<T>(ps2, 4) < 0) {
    point_reverse<T>(ps2, ps2 + 4);
  }
  ps1[4] = ps1[0];
  ps2[4] = ps2[0];
  T res = 0;
  for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
          res += intersect_area<T>(ps1[i], ps1[i+1], ps2[j], ps2[j+1]);
      }
  }
  return res;
}

}

template <typename T>
HOST_DEVICE_INLINE T
single_poly_iou(T const* const a, T const* const b) {
  float2 ps1[MAXN], ps2[MAXN];
  for (int i = 0; i < 4; i++) {
    ps1[i].x = a[i * 2];
    ps1[i].y = a[i * 2 + 1];
    ps2[i].x = b[i * 2];
    ps2[i].y = b[i * 2 + 1];
  }
  T inter_area = poly_intersection<T>(ps1, ps2);
  T union_area = fabs(area<T>(ps1, 4)) + fabs(area<T>(ps2, 4)) - inter_area;
  if (union_area == 0) {
    return (T)1;
  }
  return inter_area / union_area;
}

} // namespace pet
