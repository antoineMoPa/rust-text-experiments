// Homebrew Flash Attention — supports any d_head that fits in one CUDA block (≤ 1024).
// Uses standard CUDA — no Ampere-specific instructions.
//
// Layout: all tensors are [batch, heads, seq, d_head] row-major, f32.
//   stride_b = heads * seq * d_head
//   stride_h =         seq * d_head
//   stride_s =               d_head
//
// lse [batch, heads, seq] is written in forward and read in backward (no recomputation).

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

// ---------------------------------------------------------------------------
// Block-level helpers  (block = d_head threads, any d_head ≤ 1024)
//
// smem: scratch array of at least ceil(d_head/32) floats,
//       declared as `extern __shared__` in each kernel.
// ---------------------------------------------------------------------------

__device__ __forceinline__
float block_reduce_sum(float val, float* smem) {
    const int lane      = threadIdx.x & 31;
    const int warp_id   = threadIdx.x >> 5;
    const int num_warps = ((int)blockDim.x + 31) / 32;

    // Per-warp active-lane mask (last warp may be partial when d_head % 32 != 0).
    const int warp_threads = min(32, (int)blockDim.x - warp_id * 32);
    const unsigned int mask = (warp_threads == 32) ? 0xffffffffu
                                                   : (1u << warp_threads) - 1u;

    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);

    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < num_warps; i++) sum += smem[i];
        smem[0] = sum;
    }
    __syncthreads();
    return smem[0];
}

// Dot product of two d_head-dimensional vectors, result broadcast to all threads.
__device__ __forceinline__
float block_dot(float a, float b, float* smem) {
    return block_reduce_sum(a * b, smem);
}

// ---------------------------------------------------------------------------
// Forward kernel
//
// Grid : (batch, heads, seq)  — one block per query row
// Block: (d_head)             — one thread per head dimension
// Smem : ceil(d_head/32) floats for cross-warp reduction
// ---------------------------------------------------------------------------
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,    // [batch, heads, seq, d_head]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__       O,    // [batch, heads, seq, d_head]
    float* __restrict__       lse,  // [batch, heads, seq]
    int seq, int heads, int d_head,
    float scale, bool causal)
{
    extern __shared__ float smem[];

    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int d  = threadIdx.x;

    const int stride_b = heads * seq * d_head;
    const int stride_h =         seq * d_head;
    const int stride_s =               d_head;

    const float q_d = __ldg(&Q[b * stride_b + h * stride_h + qi * stride_s + d]);

    // Online softmax state
    float m   = -FLT_MAX;
    float l   = 0.0f;
    float o_d = 0.0f;

    const int kv_end = causal ? qi + 1 : seq;

    for (int kv = 0; kv < kv_end; kv++) {
        const int   base  = b * stride_b + h * stride_h + kv * stride_s;
        const float score = block_dot(q_d, __ldg(&K[base + d]), smem) * scale;

        const float m_new     = fmaxf(m, score);
        const float exp_score = __expf(score - m_new);
        const float rescale   = __expf(m - m_new);

        o_d = o_d * rescale + exp_score * __ldg(&V[base + d]);
        l   = l   * rescale + exp_score;
        m   = m_new;
    }

    const int out_base = b * stride_b + h * stride_h + qi * stride_s;
    O[out_base + d] = o_d / fmaxf(l, 1e-38f);

    if (d == 0)
        lse[b * heads * seq + h * seq + qi] = m + __logf(fmaxf(l, 1e-38f));
}

// ---------------------------------------------------------------------------
// Backward kernel
//
// Grid : (batch, heads, seq)  — one block per query row
// Block: (d_head)
// Smem : ceil(d_head/32) floats for cross-warp reduction
//
// dK and dV are accumulated across query rows via atomicAdd (zeroed before launch).
// dQ has no conflicts and is written directly.
// ---------------------------------------------------------------------------
__global__ void flash_attn_bwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    const float* __restrict__ lse,  // [batch, heads, seq] — saved from forward
    float* __restrict__       dQ,
    float* __restrict__       dK,   // zeroed before launch
    float* __restrict__       dV,   // zeroed before launch
    int seq, int heads, int d_head,
    float scale, bool causal)
{
    extern __shared__ float smem[];

    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int d  = threadIdx.x;

    const int stride_b = heads * seq * d_head;
    const int stride_h =         seq * d_head;
    const int stride_s =               d_head;
    const int qi_base  = b * stride_b + h * stride_h + qi * stride_s;

    const float q_d  = __ldg(&Q [qi_base + d]);
    const float o_d  = __ldg(&O [qi_base + d]);
    const float do_d = __ldg(&dO[qi_base + d]);

    // D_i = dot(O_i, dO_i) — scalar used in the softmax gradient formula
    const float D = block_dot(o_d, do_d, smem);

    const float lse_i = __ldg(&lse[b * heads * seq + h * seq + qi]);

    const int kv_end = causal ? qi + 1 : seq;

    float dq_d = 0.0f;
    for (int kv = 0; kv < kv_end; kv++) {
        const int   base  = b * stride_b + h * stride_h + kv * stride_s;
        const float k_d   = __ldg(&K[base + d]);
        const float v_d   = __ldg(&V[base + d]);

        const float score = block_dot(q_d, k_d, smem) * scale;
        const float p     = __expf(score - lse_i);

        atomicAdd(&dV[base + d], p * do_d);

        const float dp = block_dot(do_d, v_d, smem);
        const float ds = p * (dp - D);

        dq_d += ds * k_d * scale;
        atomicAdd(&dK[base + d], ds * q_d * scale);
    }

    dQ[qi_base + d] = dq_d;
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------
extern "C" {

void flash_attn_fwd(
    const float* q, const float* k, const float* v,
    float* o, float* lse,
    int batch, int seq, int heads, int d_head,
    float scale, bool causal)
{
    dim3 grid(batch, heads, seq);
    dim3 block(d_head);
    const int smem_bytes = ((d_head + 31) / 32) * sizeof(float);
    flash_attn_fwd_kernel<<<grid, block, smem_bytes>>>(
        q, k, v, o, lse, seq, heads, d_head, scale, causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "flash_attn_fwd kernel error: %s\n", cudaGetErrorString(err));
}

void flash_attn_bwd(
    const float* q, const float* k, const float* v,
    const float* o, const float* do_,
    const float* lse,
    float* dq, float* dk, float* dv,
    int batch, int seq, int heads, int d_head,
    float scale, bool causal)
{
    const size_t n = (size_t)batch * seq * heads * d_head * sizeof(float);
    cudaMemsetAsync(dk, 0, n);
    cudaMemsetAsync(dv, 0, n);

    dim3 grid(batch, heads, seq);
    dim3 block(d_head);
    const int smem_bytes = ((d_head + 31) / 32) * sizeof(float);
    flash_attn_bwd_kernel<<<grid, block, smem_bytes>>>(
        q, k, v, o, do_, lse, dq, dk, dv, seq, heads, d_head, scale, causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "flash_attn_bwd kernel error: %s\n", cudaGetErrorString(err));
}

} // extern "C"
