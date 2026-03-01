// Fused Layer Normalization CUDA kernels.
//
// Layout: x is [rows, emb] row-major f32.
//   For 3D input [batch, seq, emb]: rows = batch * seq.
//   For 2D input [batch, emb]:      rows = batch.
//
// mean and rstd [rows] are written in forward and read in backward.
//
// Block size = next_power_of_two(emb), passed by the Rust caller.
// Constraint: emb <= 1024 (enforced in Rust before launch).

#include <cuda_runtime.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xFFFFFFFFu, v, off);
    return v;
}

// Block-wide sum reduction.
// ws: caller-provided __shared__ float[32] scratch.
// All threads must call; returns the block total in every thread.
__device__ float block_sum(float v, float* ws) {
    const int tid = threadIdx.x;
    const int wid = tid >> 5;
    const int lid = tid & 31;
    const int nw  = (blockDim.x + 31) >> 5;

    v = warp_sum(v);
    if (lid == 0) ws[wid] = v;
    __syncthreads();

    float s = (tid < nw) ? ws[tid] : 0.0f;
    if (tid < 32) s = warp_sum(s);
    if (tid == 0) ws[0] = s;
    __syncthreads();

    return ws[0];
}

// ---------------------------------------------------------------------------
// Forward kernel
//
// Grid:  (rows)
// Block: (block_size) where block_size = next_power_of_two(emb) <= 1024
// ---------------------------------------------------------------------------
__global__ void layer_norm_fwd_kernel(
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ out,
    float* __restrict__ mean_out,
    float* __restrict__ rstd_out,
    int emb, float eps)
{
    __shared__ float ws[32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* xr  = x   + (size_t)row * emb;
    float*       outr = out + (size_t)row * emb;

    float xi = (tid < emb) ? __ldg(&xr[tid]) : 0.0f;

    // Mean
    float mean = block_sum(xi, ws) / (float)emb;

    // Variance
    float diff = (tid < emb) ? (xi - mean) : 0.0f;
    float rstd = rsqrtf(block_sum(diff * diff, ws) / (float)emb + eps);

    if (tid == 0) {
        mean_out[row] = mean;
        rstd_out[row] = rstd;
    }

    if (tid < emb) {
        float xhat = diff * rstd;
        outr[tid] = xhat * __ldg(&w[tid]) + __ldg(&b[tid]);
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
//
// Grid:  (rows)
// Block: (block_size)
//
// dx is written directly (no conflicts across blocks).
// dw and dv are accumulated via atomicAdd (zeroed before launch).
// ---------------------------------------------------------------------------
__global__ void layer_norm_bwd_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ mean,
    const float* __restrict__ rstd,
    float* __restrict__ dx,
    float* dw,   // no __restrict__: aliased by atomicAdd from multiple blocks
    float* db,
    int emb)
{
    // Two shared scratch arrays for two simultaneous reductions.
    __shared__ float ws1[32], ws2[32];

    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float row_mean = mean[row];
    const float row_rstd = rstd[row];
    const float inv_emb  = 1.0f / (float)emb;

    const float* xr   = x  + (size_t)row * emb;
    const float* dyr  = dy + (size_t)row * emb;
    float*       dxr  = dx + (size_t)row * emb;

    float xi   = (tid < emb) ? __ldg(&xr[tid])  : 0.0f;
    float dyi  = (tid < emb) ? __ldg(&dyr[tid]) : 0.0f;
    float wi   = (tid < emb) ? __ldg(&w[tid])   : 0.0f;

    float xhat = (xi - row_mean) * row_rstd;
    float dyw  = dyi * wi;   // upstream_grad * weight

    // Two parallel warp-level reductions.
    const int wid = tid >> 5;
    const int lid = tid & 31;
    const int nw  = (blockDim.x + 31) >> 5;

    float s1 = warp_sum(dyw);
    float s2 = warp_sum(dyw * xhat);
    if (lid == 0) { ws1[wid] = s1; ws2[wid] = s2; }
    __syncthreads();

    // First warp reduces the per-warp sums.
    float v1 = (tid < nw) ? ws1[tid] : 0.0f;
    float v2 = (tid < nw) ? ws2[tid] : 0.0f;
    if (tid < 32) { v1 = warp_sum(v1); v2 = warp_sum(v2); }
    if (tid == 0) { ws1[0] = v1; ws2[0] = v2; }
    __syncthreads();

    float sum_dyw      = ws1[0];
    float sum_dyw_xhat = ws2[0];

    if (tid < emb) {
        // dx = rstd * (dyw - mean(dyw) - xhat * mean(dyw * xhat))
        dxr[tid] = row_rstd * (dyw
                               - sum_dyw      * inv_emb
                               - xhat * sum_dyw_xhat * inv_emb);
        atomicAdd(&dw[tid], dyi * xhat);
        atomicAdd(&db[tid], dyi);
    }
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------
extern "C" {

void layer_norm_fwd(
    const float* x, const float* w, const float* b,
    float* out, float* mean_out, float* rstd_out,
    int rows, int emb, int block_size, float eps)
{
    dim3 grid(rows);
    dim3 block(block_size);
    layer_norm_fwd_kernel<<<grid, block>>>(x, w, b, out, mean_out, rstd_out, emb, eps);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "layer_norm_fwd kernel error: %s\n", cudaGetErrorString(err));
}

void layer_norm_bwd(
    const float* dy, const float* x, const float* w,
    const float* mean, const float* rstd,
    float* dx, float* dw, float* db,
    int rows, int emb, int block_size)
{
    cudaMemsetAsync(dw, 0, (size_t)emb * sizeof(float));
    cudaMemsetAsync(db, 0, (size_t)emb * sizeof(float));

    dim3 grid(rows);
    dim3 block(block_size);
    layer_norm_bwd_kernel<<<grid, block>>>(dy, x, w, mean, rstd, dx, dw, db, emb);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "layer_norm_bwd kernel error: %s\n", cudaGetErrorString(err));
}

} // extern "C"
