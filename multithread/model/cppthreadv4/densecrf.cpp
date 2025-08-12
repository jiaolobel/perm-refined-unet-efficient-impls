#include "densecrf.h"

DenseCRF::DenseCRF(int H, int W, int n_classes, int d_bifeats, int d_spfeats,
                   float theta_alpha, float theta_beta, float theta_gamma,
                   float bilateral_compat, float spatial_compat,
                   int n_iterations, int n_thread) {
    H_ = H;
    W_ = W;
    N_ = H_ * W_;
    n_classes_ = n_classes;
    d_bifeats_ = d_bifeats;
    d_spfeats_ = d_spfeats;

    theta_alpha_ = theta_alpha;
    theta_beta_ = theta_beta;
    theta_gamma_ = theta_gamma;

    bilateral_compat_ = bilateral_compat;
    spatial_compat_ = spatial_compat;

    n_iterations_ = n_iterations;

    n_thread_ = n_thread;
    if (N_ % n_thread_ != 0) {
        printf("Wrong multi-thread setting, will compute sequentially.\n");
        n_thread_ = 1;
    }
    tN_ = N_ / n_thread_;

    bilateral_lattice_ = new Permutohedral(N_, d_bifeats_, n_thread_);
    spatial_lattice_ = new Permutohedral(N_, d_spfeats, n_thread_);

    // Potts model // potts_compatibility(compatibility_matrix_);
}

DenseCRF::~DenseCRF() {
    delete bilateral_lattice_;
    delete spatial_lattice_;
}

void DenseCRF::softmax(const float *x, float *expx_shifted) {
    float *x_max = new float[N_];
    float *norm = new float[N_];
    for (int i = 0; i < N_; i++) {
        x_max[i] = x[i * n_classes_];
        norm[i] = 0;
    }

    for (int i = 0; i < N_; i++) {
        for (int j = 1; j < n_classes_; j++) {
            int cond = (x_max[i] < x[i * n_classes_ + j]);
            x_max[i] = x_max[i] * (1 - cond) + x[i * n_classes_ + j] * cond;
        }
        for (int j = 0; j < n_classes_; j++) {
            expx_shifted[i * n_classes_ + j] =
                expf(x[i * n_classes_ + j] - x_max[i]);
            norm[i] += expx_shifted[i * n_classes_ + j];
        }
        for (int j = 0; j < n_classes_; j++) {
            expx_shifted[i * n_classes_ + j] /= norm[i];
        }
    }

    delete[] x_max;
    delete[] norm;
}

void DenseCRF::seqinference(const float *unary, const float *reference,
                            float *out) {
    // Create bilateral and spatial features
    float *bilateral_feats = new float[N_ * d_bifeats_];
    float *spatial_feats = new float[N_ * d_spfeats_];

    for (int y = 0; y < H_; y++) {
        for (int x = 0; x < W_; x++) {
            int coord = y * W_ * d_bifeats_ + x * d_bifeats_;
            int refcoord = y * W_ * (d_bifeats_ - d_spfeats_) +
                           x * (d_bifeats_ - d_spfeats_);
            bilateral_feats[coord + 0] = (float)x / theta_alpha_;
            bilateral_feats[coord + 1] = (float)y / theta_alpha_;
            for (int d = d_spfeats_; d < d_bifeats_; d++) {
                bilateral_feats[coord + d] =
                    reference[refcoord + (d - d_spfeats_)] / theta_beta_;
            }

            coord = y * W_ * d_spfeats_ + x * d_spfeats_;
            spatial_feats[coord + 0] = (float)x / theta_gamma_;
            spatial_feats[coord + 1] = (float)y / theta_gamma_;
        }
    }

    // Initialize bilateral and spatial filters
    bilateral_lattice_->seqinit(bilateral_feats);
    spatial_lattice_->seqinit(spatial_feats);
    printf("Filters initialized in a sequential way.\n");

    // Free features
    delete[] bilateral_feats;
    delete[] spatial_feats;

    // Compute one-sided normalizations
    float *allOnes = new float[N_], *bilateral_norm_vals = new float[N_],
          *spatial_norm_vals = new float[N_];
    fill(allOnes, allOnes + N_, 1.f);

    bilateral_lattice_->seqcompute(allOnes, 1, false, bilateral_norm_vals);
    spatial_lattice_->seqcompute(allOnes, 1, false, spatial_norm_vals);
    delete[] allOnes;

    // Initialize Q
    float *Q = out;
    for (int i = 0; i < N_ * n_classes_; i++) {
        Q[i] = -unary[i];
    }
    softmax(Q, Q);

    float *bilateral_out = new float[N_ * n_classes_],
          *spatial_out = new float[N_ * n_classes_];

    for (int i = 0; i < n_iterations_; i++) {
        printf("Iteration %d / %d...\n", i + 1, n_iterations_);

        // Bilateral message passing
        bilateral_lattice_->seqcompute(Q, n_classes_, false, bilateral_out);

        // Spatial message passing
        spatial_lattice_->seqcompute(Q, n_classes_, false, spatial_out);

        // Dim-wise Normalization
        for (int i = 0; i < N_; i++) {
            for (int j = 0; j < n_classes_; j++) {
                int attr = i * n_classes_ + j;

                // normalization
                bilateral_out[attr] /= bilateral_norm_vals[i];
                spatial_out[attr] /= spatial_norm_vals[i];

                // Message passing
                float message_passing =
                    bilateral_compat_ * bilateral_out[attr] +
                    spatial_compat_ * spatial_out[attr];

                // Compatibility transformation
                float pairwise = compatibility_ * message_passing;

                // Local update
                Q[attr] = -unary[attr] - pairwise;
            }
        }
        // Normalize
        softmax(Q, Q); // [n_classes, N]
    }

    // release mem
    delete[] bilateral_norm_vals;
    delete[] spatial_norm_vals;
    delete[] bilateral_out;
    delete[] spatial_out;
}

void DenseCRF::tsoftmax(const float *x, float *expx_shifted) {
    float *x_max = new float[tN_];
    float *norm = new float[tN_];
    for (int i = 0; i < tN_; i++) {
        x_max[i] = x[i * n_classes_];
        norm[i] = 0;
    }

    for (int i = 0; i < tN_; i++) {
        for (int j = 1; j < n_classes_; j++) {
            int cond = (x_max[i] < x[i * n_classes_ + j]);
            x_max[i] = x_max[i] * (1 - cond) + x[i * n_classes_ + j] * cond;
        }
        for (int j = 0; j < n_classes_; j++) {
            expx_shifted[i * n_classes_ + j] =
                expf(x[i * n_classes_ + j] - x_max[i]);
            norm[i] += expx_shifted[i * n_classes_ + j];
        }
        for (int j = 0; j < n_classes_; j++) {
            expx_shifted[i * n_classes_ + j] /= norm[i];
        }
    }

    delete[] x_max;
    delete[] norm;
}

void DenseCRF::mtinference(const float *unary, const float *reference,
                           float *out) {
    // Create sequentially bilateral and spatial features
    float *bilateral_feats = new float[N_ * d_bifeats_];
    float *spatial_feats = new float[N_ * d_spfeats_];

    // == seq create feature ==
    // for (int y = 0; y < H_; y++) {
    //     for (int x = 0; x < W_; x++) {
    //         int coord = y * W_ * d_bifeats_ + x * d_bifeats_;
    //         int refcoord = y * W_ * (d_bifeats_ - d_spfeats_) +
    //                        x * (d_bifeats_ - d_spfeats_);
    //         bilateral_feats[coord + 0] = (float)x / theta_alpha_;
    //         bilateral_feats[coord + 1] = (float)y / theta_alpha_;
    //         for (int d = d_spfeats_; d < d_bifeats_; d++) {
    //             bilateral_feats[coord + d] =
    //                 reference[refcoord + (d - d_spfeats_)] / theta_beta_;
    //         }

    //         coord = y * W_ * d_spfeats_ + x * d_spfeats_;
    //         spatial_feats[coord + 0] = (float)x / theta_gamma_;
    //         spatial_feats[coord + 1] = (float)y / theta_gamma_;
    //     }
    // }

    // == mt create feature ==
    auto mtcreate_feature = [this, bilateral_feats, reference,
                             spatial_feats](int threadi) {
        int tH = H_ / n_thread_;
        int ystart = threadi * tH;
        int yend = threadi * tH + tH;
        for (int y = ystart; y < yend; y++) {
            for (int x = 0; x < W_; x++) {
                int coord = y * W_ * d_bifeats_ + x * d_bifeats_;
                int refcoord = y * W_ * (d_bifeats_ - d_spfeats_) +
                               x * (d_bifeats_ - d_spfeats_);
                bilateral_feats[coord + 0] = (float)x / theta_alpha_;
                bilateral_feats[coord + 1] = (float)y / theta_alpha_;
                for (int d = d_spfeats_; d < d_bifeats_; d++) {
                    bilateral_feats[coord + d] =
                        reference[refcoord + (d - d_spfeats_)] / theta_beta_;
                }

                coord = y * W_ * d_spfeats_ + x * d_spfeats_;
                spatial_feats[coord + 0] = (float)x / theta_gamma_;
                spatial_feats[coord + 1] = (float)y / theta_gamma_;
            }
        }
    };

    std::vector<std::thread> ftThreads;
    for (int i = 0; i < n_thread_; i++) {
        ftThreads.emplace_back(mtcreate_feature, i);
    }
    for (auto &t : ftThreads) {
        t.join();
    }

    // Initialize bilateral and spatial filters
    bilateral_lattice_->mtinit(bilateral_feats);
    spatial_lattice_->mtinit(spatial_feats);
    printf("Filters initialized in a multi-thread way.\n");

    // Delete features
    delete[] bilateral_feats;
    delete[] spatial_feats;

    // Compute one-sided normalizations
    float *allOnes = new float[N_], *bilateral_norm_vals = new float[N_],
          *spatial_norm_vals = new float[N_];
    fill(allOnes, allOnes + N_, 1.f);

    bilateral_lattice_->mtcompute(allOnes, 1, false, bilateral_norm_vals);
    spatial_lattice_->mtcompute(allOnes, 1, false, spatial_norm_vals);
    delete[] allOnes;

    // Initialize Q
    float *Q = out;

    // == multi-thread init ==
    auto tinitQ = [this](const float *tunary, float *tQ) {
        for (int i = 0; i < tN_ * n_classes_; i++) {
            tQ[i] = -tunary[i];
        }
        tsoftmax(tQ, tQ);
    };

    std::vector<std::thread> initQThreads;
    for (int i = 0; i < n_thread_; i++) {
        const float *tunary = unary + i * tN_ * n_classes_;
        float *tQ = Q + i * tN_ * n_classes_;
        initQThreads.emplace_back(tinitQ, tunary, tQ);
    }
    for (auto &t : initQThreads) {
        t.join();
    }

    float *bilateral_out = new float[N_ * n_classes_],
          *spatial_out = new float[N_ * n_classes_];

    for (int i = 0; i < n_iterations_; i++) {
        printf("Iteration %d / %d...\n", i + 1, n_iterations_);

        // Bilateral message passing
        bilateral_lattice_->mtcompute(Q, n_classes_, false, bilateral_out);

        // Spatial message passing
        spatial_lattice_->mtcompute(Q, n_classes_, false, spatial_out);

        // == Multi-thread dim-wise normalization ==
        auto tnormDim = [this](const float *tbilateral_out,
                              const float *tspatial_out,
                              const float *tbilateral_norm_vals,
                              const float *tspatial_norm_vals,
                              const float *tunary, float *tQ) {
            for (int i = 0; i < tN_; i++) {
                for (int j = 0; j < n_classes_; j++) {
                    int idx = i * n_classes_ + j;
                    float norm_bilateral_out =
                        tbilateral_out[idx] / tbilateral_norm_vals[i];
                    float norm_spatial_out =
                        tspatial_out[idx] / tspatial_norm_vals[i];

                    // Normalization and Message passing
                    float message_passing =
                        bilateral_compat_ * norm_bilateral_out +
                        spatial_compat_ * norm_spatial_out;

                    // Compatibility transformation
                    float pairwise = compatibility_ * message_passing;

                    // Local update
                    tQ[idx] = -tunary[idx] - pairwise;
                }
            }
            tsoftmax(tQ, tQ);
        };

        std::vector<std::thread> normThreads;
        for (int i = 0; i < n_thread_; i++) {
            const float *tbilateral_out = bilateral_out + i * tN_ * n_classes_;
            const float *tspatial_out = spatial_out + i * tN_ * n_classes_;
            const float *tbilateral_norm_vals = bilateral_norm_vals + i * tN_;
            const float *tspatial_norm_vals = spatial_norm_vals + i * tN_;
            const float *tunary = unary + i * tN_ * n_classes_;
            float *tQ = Q + i * tN_ * n_classes_;
            normThreads.emplace_back(tnormDim, tbilateral_out, tspatial_out,
                                     tbilateral_norm_vals, tspatial_norm_vals,
                                     tunary, tQ);
        }
        for (auto &t : normThreads) {
            t.join();
        }
    }

    // release mem
    delete[] bilateral_norm_vals;
    delete[] spatial_norm_vals;
    delete[] bilateral_out;
    delete[] spatial_out;
}
