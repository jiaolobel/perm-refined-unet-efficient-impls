#ifndef PERMUTOHEDRAL_H
#define PERMUTOHEDRAL_H

#include <iostream>
#include <stdbool.h>
#include <string.h>
#include <vector>

#include <thread>

using namespace std;

class Permutohedral {
  private:
    int N_ = 0, d_ = 0, d1_ = 0, M_ = 0;
    float alpha_ = 0.f;

    int n_thread_ = 1;
    int tN_ = 0;

    int *os_ = NULL;
    float *ws_ = NULL;

    int *blur_neighbors_ = NULL;

  public:
    Permutohedral(int N, int d, int n_thread);
    void seqinit(const float *feature);
    void tinit(const float *tfeature, int threadi, short *tkey);
    void mtinit(const float *feature);
    void delete_blurneighbors_();
    void seqcompute(const float *inp, const int value_size, const bool reversal,
                    float *out);
    void mtcompute(const float *inp, const int value_size, const bool reversal,
                   float *out);
    ~Permutohedral();
};

#endif