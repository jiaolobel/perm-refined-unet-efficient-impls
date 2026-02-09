# distutils: language = c++
# cython: language_level = 3
# distutils: sources = [densecrf.cpp, permutohedral.cpp]

cdef extern from "densecrf.h":
    cdef cppclass DenseCRF:
        DenseCRF(int H, int W, int n_classes, int d_bifeats, int d_spfeats, float theta_alpha, float theta_beta, float theta_gamma, float bilateral_compat, float spatial_compat, int n_iterations, int n_thread) except +

        void seqinference(const float *unary, const float *reference, float *out) except +

        void mtinference(const float *unary, const float *reference, float *out) except +

cdef class PyDenseCRF:
    cdef DenseCRF *dcrf

    def __cinit__(self, int H, int W, int n_classes, int d_bifeats, int d_spfeats, float theta_alpha, float theta_beta, float theta_gamma, float bilateral_compat, float spatial_compat, int n_iterations, int n_thread):
        self.dcrf = new DenseCRF(H, W, n_classes, d_bifeats, d_spfeats, theta_alpha, theta_beta, theta_gamma, bilateral_compat, spatial_compat, n_iterations, n_thread)

    def seqinference(self, float[:] unary, float[:] reference, float[:] out):
        self.dcrf.seqinference(&unary[0], &reference[0], &out[0])

    def mtinference(self, float[:] unary, float[:] reference, float[:] out):
        self.dcrf.mtinference(&unary[0], &reference[0], &out[0])

    def __dealloc__(self):
        del self.dcrf