# --- INICIO BLOQUE GENERADO CON IA ---
# Implementación Cython desarrollada con asistencia de IA.



import numpy as np
cimport numpy as np

def im2col_forward_cython(
    np.ndarray[np.float32_t, ndim=4] input,
    np.ndarray[np.float32_t, ndim=4] kernels,
    np.ndarray[np.float32_t, ndim=1] biases,
    int stride,
    int padding
):
    cdef int batch_size = input.shape[0]
    cdef int in_c = input.shape[1]
    cdef int in_h = input.shape[2]
    cdef int in_w = input.shape[3]
    cdef int out_channels = kernels.shape[0]
    cdef int k_h = kernels.shape[2]
    cdef int k_w = kernels.shape[3]
    cdef int out_h, out_w
    cdef int b, oc, ic, i, j, kh, kw
    cdef float val

    if padding > 0:
        input = np.pad(input,
                       ((0,0),(0,0),(padding,padding),(padding,padding)),
                       mode='constant').astype(np.float32)

    out_h = (input.shape[2] - k_h) // stride + 1
    out_w = (input.shape[3] - k_w) // stride + 1

    cdef np.ndarray[np.float32_t, ndim=4] output = np.zeros(
        (batch_size, out_channels, out_h, out_w), dtype=np.float32)

    for b in range(batch_size):
        for oc in range(out_channels):
            for ic in range(in_c):
                for i in range(out_h):
                    for j in range(out_w):
                        val = 0.0
                        for kh in range(k_h):
                            for kw in range(k_w):
                                val += (input[b, ic, i*stride+kh, j*stride+kw]
                                        * kernels[oc, ic, kh, kw])
                        output[b, oc, i, j] += val
            output[b, oc] += biases[oc]

    return output

# --- FIN BLOQUE GENERADO CON IA ---