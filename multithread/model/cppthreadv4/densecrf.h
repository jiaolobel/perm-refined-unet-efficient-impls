#ifndef DENSECRF_H
#define DENSECRF_H

#include <algorithm>
#include <iostream>
#include <math.h>
#include <stdbool.h>
#include <string.h>
#include <vector>

#include "permutohedral.h"

class DenseCRF {
  private:
  protected:
    int H_, W_, N_, n_classes_, d_bifeats_, d_spfeats_;

    float theta_alpha_, theta_beta_, theta_gamma_;

    float bilateral_compat_, spatial_compat_;

    int n_iterations_;

    int n_thread_ = 1;

    int tN_ = 0;

    Permutohedral *bilateral_lattice_ = NULL, *spatial_lattice_ = NULL;

    float compatibility_ = -1.f;

  public:
    DenseCRF(int H, int W, int n_classes, int d_bifeats, int d_spfeats,
             float theta_alpha, float theta_beta, float theta_gamma,
             float bilateral_compat, float spatial_compat, int n_iterations,
             int n_thread);

    ~DenseCRF();

    void softmax(const float *x, float *expx_shifted);

    void seqinference(const float *unary, const float *reference, float *out);

    void tsoftmax(const float *x, float *expx_shifted);

    void mtinference(const float *unary, const float *reference, float *out);
};

#endif