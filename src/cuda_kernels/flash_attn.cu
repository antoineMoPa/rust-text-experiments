// Homebrew Flash Attention for sm_75 (GTX 1660 / Turing).
// Uses standard CUDA — no Ampere-specific instructions.
//
// Layout: all tensors are [batch, seq, heads, d_head] row-major, f32.
// lse (log-sum-exp) is [batch, heads, seq], saved in forward for backward.

#include <cuda_runtime.h>
#include <float.h>

// ---------------------------------------------------------------------------
// Warp helpers  (d_head <= 32, so everything fits in one warp)
// ---------------------------------------------------------------------------

__device__ __forceinline__
float warp_reduce_sum(float val, unsigned int mask) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

// Dot product of two d_head-dimensional vectors, one element per thread.
// Result is broadcast to all threads in the warp.
__device__ __forceinline__
float warp_dot(float a, float b, unsigned int mask) {
    return __shfl_sync(mask, warp_reduce_sum(a * b, mask), 0);
}

// ---------------------------------------------------------------------------
// Forward kernel
//
// Grid : (batch, heads, seq)  — one block per query row
// Block: (d_head)             — one thread per head dimension
// ---------------------------------------------------------------------------
__global__ void flash_attn_fwd_kernel(
    const float* __restrict__ Q,    // [batch, seq, heads, d_head]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__       O,    // [batch, seq, heads, d_head]
    float* __restrict__       lse,  // [batch, heads, seq]
    int seq, int heads, int d_head,
    float scale, bool causal)
{
    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int d  = threadIdx.x;

    // Active-lane mask for warp ops (safe for d_head <= 32)
    const unsigned int mask = (d_head == 32) ? 0xffffffffu : (1u << d_head) - 1u;

    const int stride_b = seq   * heads * d_head;
    const int stride_s =         heads * d_head;
    const int stride_h =                 d_head;

    const float q_d = Q[b * stride_b + qi * stride_s + h * stride_h + d];

    // Online softmax state
    float m   = -FLT_MAX;  // running max
    float l   = 0.0f;      // running sum of exp(score - m)
    float o_d = 0.0f;      // running weighted-value accumulator

    const int kv_end = causal ? qi + 1 : seq;

    for (int kv = 0; kv < kv_end; kv++) {
        const int   base  = b * stride_b + kv * stride_s + h * stride_h;
        const float score = warp_dot(q_d, K[base + d], mask) * scale;

        const float m_new     = fmaxf(m, score);
        const float exp_score = expf(score - m_new);
        // When m == -FLT_MAX (first iter), expf(-FLT_MAX - m_new) underflows to 0 — correct.
        const float rescale   = expf(m - m_new);

        o_d = o_d * rescale + exp_score * V[base + d];
        l   = l   * rescale + exp_score;
        m   = m_new;
    }

    // Write output and save log-sum-exp for backward
    const int out_base = b * stride_b + qi * stride_s + h * stride_h;
    O[out_base + d] = o_d / fmaxf(l, 1e-38f);

    if (d == 0) {
        lse[b * heads * seq + h * seq + qi] = m + logf(fmaxf(l, 1e-38f));
    }
}

// ---------------------------------------------------------------------------
// Backward kernel
//
// Grid : (batch, heads, seq)  — one block per query row
// Block: (d_head)
//
// dK and dV are accumulated across query rows via atomicAdd (initialized to
// zero before launch). dQ has no conflicts and is written directly.
//
// lse is not passed in — we recompute it to avoid storing an extra tensor.
// This costs one extra O(seq) pass but keeps the interface simple.
// ---------------------------------------------------------------------------
__global__ void flash_attn_bwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ O,
    const float* __restrict__ dO,
    float* __restrict__       dQ,
    float* __restrict__       dK,   // zeroed before launch
    float* __restrict__       dV,   // zeroed before launch
    int seq, int heads, int d_head,
    float scale, bool causal)
{
    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int d  = threadIdx.x;

    const unsigned int mask = (d_head == 32) ? 0xffffffffu : (1u << d_head) - 1u;

    const int stride_b = seq   * heads * d_head;
    const int stride_s =         heads * d_head;
    const int stride_h =                 d_head;
    const int qi_base  = b * stride_b + qi * stride_s + h * stride_h;

    const float q_d  = Q [qi_base + d];
    const float o_d  = O [qi_base + d];
    const float do_d = dO[qi_base + d];

    // D_i = dot(O_i, dO_i) — scalar used in the softmax gradient formula
    const float D = warp_dot(o_d, do_d, mask);

    const int kv_end = causal ? qi + 1 : seq;

    // Pass 1: recompute lse_i (log-sum-exp of scores for query row qi)
    float m = -FLT_MAX, l = 0.0f;
    for (int kv = 0; kv < kv_end; kv++) {
        const int   base  = b * stride_b + kv * stride_s + h * stride_h;
        const float score = warp_dot(q_d, K[base + d], mask) * scale;
        const float m_new = fmaxf(m, score);
        l = l * expf(m - m_new) + expf(score - m_new);
        m = m_new;
    }
    const float lse_i = m + logf(fmaxf(l, 1e-38f));

    // Pass 2: compute dQ, and accumulate dK / dV
    float dq_d = 0.0f;
    for (int kv = 0; kv < kv_end; kv++) {
        const int   base  = b * stride_b + kv * stride_s + h * stride_h;
        const float k_d   = K[base + d];
        const float v_d   = V[base + d];

        const float score = warp_dot(q_d, k_d, mask) * scale;
        const float p     = expf(score - lse_i);   // softmax weight p_ij

        // dV_kv += p_ij * dO_qi
        atomicAdd(&dV[base + d], p * do_d);

        // dp_ij = dot(dO_qi, V_kv)
        const float dp = warp_dot(do_d, v_d, mask);

        // ds_ij = p_ij * (dp_ij - D_i)   [softmax Jacobian]
        const float ds = p * (dp - D);

        // dQ_qi += ds_ij * K_kv * scale
        dq_d += ds * k_d * scale;

        // dK_kv += ds_ij * Q_qi * scale
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
    flash_attn_fwd_kernel<<<grid, block>>>(
        q, k, v, o, lse, seq, heads, d_head, scale, causal);
}

void flash_attn_bwd(
    const float* q, const float* k, const float* v,
    const float* o, const float* do_,
    float* dq, float* dk, float* dv,
    int batch, int seq, int heads, int d_head,
    float scale, bool causal)
{
    const size_t n = (size_t)batch * seq * heads * d_head * sizeof(float);
    cudaMemsetAsync(dk, 0, n);
    cudaMemsetAsync(dv, 0, n);

    dim3 grid(batch, heads, seq);
    dim3 block(d_head);
    flash_attn_bwd_kernel<<<grid, block>>>(
        q, k, v, o, do_, dq, dk, dv, seq, heads, d_head, scale, causal);
}

} // extern "C"
