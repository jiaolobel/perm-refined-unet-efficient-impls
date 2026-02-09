/*
    Copyright (c) 2013, Philipp Krähenbühl, Libin Jiao
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl and Libin Jiao ''AS IS'' AND
   ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
   THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// final version, public available at github (OA)

// 2025.05.14 try to remove if
// 2025.07.31 try multi-processing
// 2025.08.06 try thread in cpp

#include "permutohedral.h"

class HashTable {
  protected:
    size_t key_size_, filled_, capacity_;
    std::vector<short> keys_;
    std::vector<int> table_;
    void grow() {
        // Create the new memory and copy the values in
        int old_capacity = capacity_;
        capacity_ *= 2;
        std::vector<short> old_keys((old_capacity + 10) * key_size_);
        std::copy(keys_.begin(), keys_.end(), old_keys.begin());
        std::vector<int> old_table(capacity_, -1);

        // Swap the memory
        table_.swap(old_table);
        keys_.swap(old_keys);

        // Reinsert each element
        for (int i = 0; i < old_capacity; i++)
            if (old_table[i] >= 0) {
                int e = old_table[i];
                size_t h = hash(getKey(e)) % capacity_;
                for (; table_[h] >= 0; h = h < capacity_ - 1 ? h + 1 : 0)
                    ;
                table_[h] = e;
            }
    }
    size_t hash(const short *k) {
        size_t r = 0;
        for (size_t i = 0; i < key_size_; i++) {
            r += k[i];
            r *= 1664525;
        }
        return r;
    }

  public:
    explicit HashTable(int key_size, int n_elements)
        : key_size_(key_size), filled_(0), capacity_(2 * n_elements),
          keys_((capacity_ / 2 + 10) * key_size_), table_(2 * n_elements, -1) {}
    int size() const { return filled_; }
    void reset() {
        filled_ = 0;
        std::fill(table_.begin(), table_.end(), -1);
    }
    int find(const short *k, bool create = false) {
        if (2 * filled_ >= capacity_)
            grow();
        // Get the hash value
        size_t h = hash(k) % capacity_;
        // Find the element with he right key, using linear probing
        while (1) {
            int e = table_[h];
            if (e == -1) {
                if (create) {
                    // Insert a new key and return the new id
                    for (size_t i = 0; i < key_size_; i++)
                        keys_[filled_ * key_size_ + i] = k[i];
                    return table_[h] = filled_++;
                } else
                    return -1;
            }
            // Check if the current key is The One
            bool good = true;
            for (size_t i = 0; i < key_size_ && good; i++)
                if (keys_[e * key_size_ + i] != k[i])
                    good = false;
            if (good)
                return e;
            // Continue searching
            h++;
            if (h == capacity_)
                h = 0;
        }
    }
    const short *getKey(int i) const { return &keys_[i * key_size_]; }
};

Permutohedral::Permutohedral(int N, int d, int n_thread) {
    N_ = N;
    d_ = d;
    n_thread_ = n_thread;
    if (N_ % n_thread_ != 0) {
        printf("Wrong multi-thread setting, will compute sequentially.\n");
        n_thread_ = 1;
    }
    tN_ = N_ / n_thread_;

    d1_ = d + 1;

    // For slicing
    alpha_ = 1.f / (1.f + powf(2, -d_));

    os_ = new int[N_ * d1_];   // channel-last
    ws_ = new float[N_ * d1_]; // channel-last
}

void Permutohedral::seqinit(const float *feature) {
    HashTable table(d_, N_ * d1_);
    // Compute the lattice coordinates for each feature [there is going to
    // be a lot of magic here]

    // Allocate local memories
    float *elevated = new float[d1_];
    int *rem0 = new int[d1_];
    float *barycentric = new float[d_ + 2];
    short *rank = new short[d1_];
    short *key = new short[d_];
    short *canonical = new short[d1_ * d1_];
    float *scale_factor = new float[d_];

    // Compute the canonical simplex
    for (int i = 0; i < d1_; i++) {
        for (int j = 0; j < d1_ - i; j++) {
            canonical[i * d1_ + j] = i;
        }
        for (int j = d1_ - i; j < d1_; j++) {
            canonical[i * d1_ + j] = i - d1_;
        }
    }

    // Expected standard deviation of our filter (p.6 in [Adams et al.,
    // 2010])
    float inv_std_dev = sqrtf(2. / 3.) * d1_;
    // Compute the diagonal part of E (p.5 in [Adams et al., 2010])
    for (int i = 0; i < d_; i++) {
        scale_factor[i] = 1.f / sqrtf((i + 2) * (i + 1)) * inv_std_dev;
    }

    // Compute the simplex each feature lies in
    for (int n = 0; n < N_; n++) {
        // Elevate the feature (y = Ep, see p.5 in [Adams et al., 2010])
        const float *f = &feature[n * d_];

        // sm contains the sum of 1...n of our feature vector
        float sm = 0;
        for (int j = d_; j > 0; j--) {
            float cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - j * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.f / d1_;
        float up_factor = d1_;
        int sum = 0;
        for (int i = 0; i < d1_; i++) {
            int rd2;
            float v = down_factor * elevated[i];
            float up = ceilf(v) * up_factor;
            float down = floorf(v) * up_factor;

            short cond = (up - elevated[i] < elevated[i] - down);
            rd2 = (short)up * cond + (short)down * (1 - cond);

            rem0[i] = rd2;
            sum += rd2 * down_factor;
        }

        // Find the simplex we are in and store it in rank (where rank
        // describes what position coordinate i has in the sorted order of
        // the feature values)
        fill(rank, rank + d1_, 0);
        for (int i = 0; i < d_; i++) {
            double di = elevated[i] - rem0[i];
            for (int j = i + 1; j < d1_; j++) {
                short cond = (elevated[i] - rem0[i]) < (elevated[j] - rem0[j]);
                rank[i] += 1 * cond;
                rank[j] += 1 * (1 - cond);
            }
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i < d1_; i++) {
            rank[i] += sum;

            short cond1 = rank[i] < 0;
            short cond2 = rank[i] > d_;
            rank[i] += d1_ * cond1 - d1_ * cond2;
            rem0[i] += d1_ * cond1 - d1_ * cond2;
        }

        // Compute the barycentric coordinates (p.10 in [Adams et al.,
        // 2010])
        fill(barycentric, barycentric + d_ + 2, 0);
        for (int i = 0; i < d1_; i++) {
            float v = (elevated[i] - rem0[i]) * down_factor;
            barycentric[d_ - rank[i]] += v;
            barycentric[d1_ - rank[i]] -= v;
        }
        // Wrap around
        barycentric[0] += 1.f + barycentric[d1_];

        // Compute all vertices and their offset
        for (int remainder = 0; remainder < d1_; remainder++) {
            for (int i = 0; i < d_; i++) {
                key[i] = rem0[i] + canonical[remainder * d1_ + rank[i]];
            }

            os_[n * d1_ + remainder] = table.find(key, true) + 1;
            ws_[n * d1_ + remainder] = barycentric[remainder];
        }
    }

    // Delete
    delete[] elevated;
    delete[] rem0;
    delete[] barycentric;
    delete[] rank;
    delete[] key;
    delete[] canonical;
    delete[] scale_factor;

    // Find the neighbors of each lattice point

    // Get the number of vertices in the lattice
    M_ = table.size();

    if (blur_neighbors_ != NULL) {
        delete[] blur_neighbors_;
    }
    blur_neighbors_ = new int[d1_ * M_ * 2];

    short *n1 = new short[d1_];
    short *n2 = new short[d1_];

    // For each of d + 1 axes,
    for (int j = 0; j < d1_; j++) {
        for (int i = 0; i < M_; i++) {
            const short *key = table.getKey(i);

            for (int k = 0; k < d_; k++) {
                n1[k] = key[k] - 1;
                n2[k] = key[k] + 1;
            }
            n1[j] = key[j] + d_;
            n2[j] = key[j] - d_;

            blur_neighbors_[j * M_ * 2 + i * 2 + 0] = table.find(n1) + 1;
            blur_neighbors_[j * M_ * 2 + i * 2 + 1] = table.find(n2) + 1;
        }
    }

    delete[] n1;
    delete[] n2;
}

void Permutohedral::tinit(const float *tfeature, int threadi, short *tkey) {
    // Compute the lattice coordinates for each feature [there is going to
    // be a lot of magic here]

    float *tw = ws_ + threadi * tN_ * d1_;

    // Allocate local memories
    float *elevated = new float[d1_];
    int *rem0 = new int[d1_];
    float *barycentric = new float[d_ + 2];
    short *rank = new short[d1_];
    short *canonical = new short[d1_ * d1_];
    float *scale_factor = new float[d_];

    // Compute the canonical simplex
    for (int i = 0; i < d1_; i++) {
        for (int j = 0; j < d1_ - i; j++) {
            canonical[i * d1_ + j] = i;
        }
        for (int j = d1_ - i; j < d1_; j++) {
            canonical[i * d1_ + j] = i - d1_;
        }
    }

    // Expected standard deviation of our filter (p.6 in [Adams et al.,
    // 2010])
    float inv_std_dev = sqrtf(2. / 3.) * d1_;
    // Compute the diagonal part of E (p.5 in [Adams et al., 2010])
    for (int i = 0; i < d_; i++) {
        scale_factor[i] = 1.f / sqrtf((i + 2) * (i + 1)) * inv_std_dev;
    }

    // Compute the simplex each feature lies in
    for (int n = 0; n < tN_; n++) {
        // Elevate the feature (y = Ep, see p.5 in [Adams et al., 2010])
        const float *f = &tfeature[n * d_];
        // float *barycentric = &pbarycentric[n * (d_ + 2)];
        short *k = &tkey[n * d1_ * d_];

        // sm contains the sum of 1...n of our feature vector
        float sm = 0;
        for (int j = d_; j > 0; j--) {
            float cf = f[j - 1] * scale_factor[j - 1];
            elevated[j] = sm - j * cf;
            sm += cf;
        }
        elevated[0] = sm;

        // Find the closest 0-colored simplex through rounding
        float down_factor = 1.f / d1_;
        float up_factor = d1_;
        int sum = 0;
        for (int i = 0; i < d1_; i++) {
            int rd2;
            float v = down_factor * elevated[i];
            float up = ceilf(v) * up_factor;
            float down = floorf(v) * up_factor;

            short cond = (up - elevated[i] < elevated[i] - down);
            rd2 = (short)up * cond + (short)down * (1 - cond);

            rem0[i] = rd2;
            sum += rd2 * down_factor;
        }

        // Find the simplex we are in and store it in rank (where rank
        // describes what position coordinate i has in the sorted order of
        // the feature values)
        fill(rank, rank + d1_, 0);
        for (int i = 0; i < d_; i++) {
            double di = elevated[i] - rem0[i];
            for (int j = i + 1; j < d1_; j++) {
                short cond = (elevated[i] - rem0[i]) < (elevated[j] - rem0[j]);
                rank[i] += 1 * cond;
                rank[j] += 1 * (1 - cond);
            }
        }

        // If the point doesn't lie on the plane (sum != 0) bring it back
        for (int i = 0; i < d1_; i++) {
            rank[i] += sum;

            short cond1 = rank[i] < 0;
            short cond2 = rank[i] > d_;
            rank[i] += d1_ * cond1 - d1_ * cond2;
            rem0[i] += d1_ * cond1 - d1_ * cond2;
        }

        // Compute the barycentric coordinates (p.10 in [Adams et al.,
        // 2010])
        fill(barycentric, barycentric + d_ + 2, 0);
        for (int i = 0; i < d1_; i++) {
            float v = (elevated[i] - rem0[i]) * down_factor;
            barycentric[d_ - rank[i]] += v;
            barycentric[d1_ - rank[i]] -= v;
        }
        // Wrap around
        barycentric[0] += 1.f + barycentric[d1_];

        // Compute all vertices and their offset
        for (int remainder = 0; remainder < d1_; remainder++) {
            for (int i = 0; i < d_; i++) {
                k[remainder * d_ + i] =
                    rem0[i] + canonical[remainder * d1_ + rank[i]];
            }
            tw[n * d1_ + remainder] = barycentric[remainder];
        }
    }

    // Delete
    delete[] elevated;
    delete[] rem0;
    delete[] barycentric;
    delete[] rank;
    delete[] canonical;
    delete[] scale_factor;
}

void Permutohedral::mtinit(const float *feature) {
    short *key = new short[N_ * d1_ * d_];

    std::vector<std::thread> threads;
    auto call_tinit = [this](const float *tfeature, int threadi, short *tkey) { tinit(tfeature, threadi, tkey); };

    for (int i = 0; i < n_thread_; i++) {
        const float *tfeature = feature + i * tN_ * d_;
        short *tkey = key + i * tN_ * d1_ * d_;
        threads.emplace_back(call_tinit, tfeature, i, tkey);
    }

    for (auto &t : threads) {
        t.join();
    }

    HashTable table(d_, N_ * d1_);

    for (int n = 0; n < N_; n++) {
        for (int remainder = 0; remainder < d1_; remainder++) {
            os_[n * d1_ + remainder] =
                table.find(key + (n * d1_ + remainder) * d_, true) + 1;
        }
    }

    delete[] key;

    // Get the number of vertices in the lattice
    M_ = table.size();

    if (blur_neighbors_ != NULL) {
        delete[] blur_neighbors_;
    }
    blur_neighbors_ = new int[d1_ * M_ * 2];

    short *n1 = new short[d1_];
    short *n2 = new short[d1_];

    // For each of d + 1 axes,
    for (int j = 0; j < d1_; j++) {
        for (int i = 0; i < M_; i++) {
            const short *key = table.getKey(i);

            for (int k = 0; k < d_; k++) {
                n1[k] = key[k] - 1;
                n2[k] = key[k] + 1;
            }
            n1[j] = key[j] + d_;
            n2[j] = key[j] - d_;

            blur_neighbors_[j * M_ * 2 + i * 2 + 0] = table.find(n1) + 1;
            blur_neighbors_[j * M_ * 2 + i * 2 + 1] = table.find(n2) + 1;
        }
    }

    delete[] n1;
    delete[] n2;
}

void Permutohedral::delete_blurneighbors_() {
    delete[] blur_neighbors_;
    blur_neighbors_ = NULL;
}

void Permutohedral::seqcompute(const float *inp, const int value_size,
                               const bool reversal, float *out) {
    float *value = new float[(M_ + 2) * value_size];
    float *buffer = new float[(M_ + 2) * value_size];
    fill(value, value + (M_ + 2) * value_size, 0.f);
    fill(buffer, buffer + (M_ + 2) * value_size, 0.f);

    // Splatting
    for (int i = 0; i < N_; i++) {
        for (int j = 0; j < d1_; j++) {
            int o = os_[i * d1_ + j];
            float w = ws_[i * d1_ + j];

            for (int v = 0; v < value_size; v++) {
                value[o * value_size + v] += w * inp[i * value_size + v];
            }
        }
    }

    // Blurring
    for (int j = reversal ? d_ : 0; j <= d_ && j >= 0; reversal ? j-- : j++) {
        for (int i = 0; i < M_; i++) {
            int n1 = blur_neighbors_[j * M_ * 2 + i * 2 + 0];
            int n2 = blur_neighbors_[j * M_ * 2 + i * 2 + 1];
            for (int v = 0; v < value_size; v++) {
                buffer[(i + 1) * value_size + v] =
                    value[(i + 1) * value_size + v] +
                    0.5f * (value[n1 * value_size + v] +
                            value[n2 * value_size + v]);
            }
        }

        // results should be used for the next convolution.
        float *tmp = buffer;
        buffer = value;
        value = tmp;
    }

    fill(out, out + N_ * value_size, 0);

    for (int i = 0; i < N_; i++) {
        for (int j = 0; j < d1_; j++) {
            int o = os_[i * d1_ + j];
            float w = ws_[i * d1_ + j];
            for (int v = 0; v < value_size; v++) {
                out[i * value_size + v] +=
                    w * value[o * value_size + v] * alpha_;
            }
        }
    }

    delete[] value;
    delete[] buffer;
}

void Permutohedral::mtcompute(const float *inp, const int value_size,
                              const bool reversal, float *out) {
    float *value = new float[(M_ + 2) * value_size];
    float *buffer = new float[(M_ + 2) * value_size];
    fill(value, value + (M_ + 2) * value_size, 0.f);
    fill(buffer, buffer + (M_ + 2) * value_size, 0.f);

    // Multi-thread splatting
    float *tvalues = new float[n_thread_ * (M_ + 2) * value_size];
    fill(tvalues, tvalues + n_thread_ * (M_ + 2) * value_size, 0.f);

    auto tsplat = [this, inp, value_size, tvalues](int threadi) {
        const float *tinp = inp + threadi * tN_ * value_size;
        float *tvalue = tvalues + threadi * (M_ + 2) * value_size;
        int *tos = os_ + threadi * tN_ * d1_;
        float *tws = ws_ + threadi * tN_ * d1_;
        for (int i = 0; i < tN_; i++) {
            for (int j = 0; j < d1_; j++) {
                int o = tos[i * d1_ + j];
                float w = tws[i * d1_ + j];

                for (int v = 0; v < value_size; v++) {
                    tvalue[o * value_size + v] += w * tinp[i * value_size + v];
                }
            }
        }
    };

    std::vector<std::thread> splatThreads;
    for (int i = 0; i < n_thread_; i++) {
        splatThreads.emplace_back(tsplat, i);
    }
    for (auto &t : splatThreads) {
        t.join();
    }

    // Reduce
    for (int i = 0; i < n_thread_; i++) {
        float *tvalue = tvalues + i * (M_ + 2) * value_size;
        for (int j = 0; j < M_ + 2; j++) {
            for (int v = 0; v < value_size; v++) {
                value[j * value_size + v] += tvalue[j * value_size + v];
            }
        }
    }

    delete[] tvalues;

    // Blurring
    for (int j = reversal ? d_ : 0; j <= d_ && j >= 0; reversal ? j-- : j++) {
        for (int i = 0; i < M_; i++) {
            int n1 = blur_neighbors_[j * M_ * 2 + i * 2 + 0];
            int n2 = blur_neighbors_[j * M_ * 2 + i * 2 + 1];
            for (int v = 0; v < value_size; v++) {
                buffer[(i + 1) * value_size + v] =
                    value[(i + 1) * value_size + v] +
                    0.5f * (value[n1 * value_size + v] +
                            value[n2 * value_size + v]);
            }
        }

        // results should be used for the next convolution.
        float *tmp = buffer;
        buffer = value;
        value = tmp;
    }

    // Seq slicing
    fill(out, out + N_ * value_size, 0);

    // Multi-thread splicing
    auto tslice = [this, out, value_size, value](int threadi) {
        float *tout = out + threadi * tN_ * value_size;
        int *tos = os_ + threadi * tN_ * d1_;
        float *tws = ws_ + threadi * tN_ * d1_;

        for (int i = 0; i < tN_; i++) {
            for (int j = 0; j < d1_; j++) {
                int o = tos[i * d1_ + j];
                float w = tws[i * d1_ + j];
                for (int v = 0; v < value_size; v++) {
                    tout[i * value_size + v] +=
                        w * value[o * value_size + v] * alpha_;
                }
            }
        }
    };

    std::vector<std::thread> sliceThreads;
    for (int i = 0; i < n_thread_; i++) {
        sliceThreads.emplace_back(tslice, i);
    }
    for (auto &t : sliceThreads) {
        t.join();
    }

    delete[] value;
    delete[] buffer;
}

Permutohedral::~Permutohedral() {
    delete[] os_;
    delete[] ws_;
    if (blur_neighbors_ != NULL) {
        delete[] blur_neighbors_;
    }
}
