// Homebrew Flash Attention for sm_75 (GTX 1660 / Turing).
// Uses standard CUDA — no Ampere-specific instructions.
//
// Layout: all tensors are [batch, heads, seq, d_head] row-major, f32.
//   stride_b = heads * seq * d_head
//   stride_h =         seq * d_head
//   stride_s =               d_head    <-- inner-loop stride is d_head (9 floats),
//                                          giving ~3-4 kv rows per 128-byte cache line.
//
// lse [batch, heads, seq] is written in forward and read in backward (no recomputation).

#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>

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
    const float* __restrict__ Q,    // [batch, heads, seq, d_head]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__       O,    // [batch, heads, seq, d_head]
    float* __restrict__       lse,  // [batch, heads, seq]
    int seq, int heads, int d_head,
    float scale, bool causal)
{
    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int d  = threadIdx.x;

    // Active-lane mask for warp ops (d_head <= 32 enforced by caller).
    // NOTE: (1u << 32) is UB; the ternary special-cases d_head == 32 to avoid it.
    const unsigned int mask = (d_head == 32) ? 0xffffffffu : (1u << d_head) - 1u;

    // [batch, heads, seq, d_head] strides — inner-loop step is d_head, not heads*d_head.
    const int stride_b = heads * seq * d_head;
    const int stride_h =         seq * d_head;
    const int stride_s =               d_head;

    const float q_d = __ldg(&Q[b * stride_b + h * stride_h + qi * stride_s + d]);

    // Online softmax state
    float m   = -FLT_MAX;  // running max
    float l   = 0.0f;      // running sum of exp(score - m)
    float o_d = 0.0f;      // running weighted-value accumulator

    const int kv_end = causal ? qi + 1 : seq;

    for (int kv = 0; kv < kv_end; kv++) {
        const int   base  = b * stride_b + h * stride_h + kv * stride_s;
        const float score = warp_dot(q_d, __ldg(&K[base + d]), mask) * scale;

        const float m_new     = fmaxf(m, score);
        const float exp_score = __expf(score - m_new);
        // When m == -FLT_MAX (first iter), __expf(-FLT_MAX - m_new) underflows to 0 — correct.
        const float rescale   = __expf(m - m_new);

        o_d = o_d * rescale + exp_score * __ldg(&V[base + d]);
        l   = l   * rescale + exp_score;
        m   = m_new;
    }

    const int out_base = b * stride_b + h * stride_h + qi * stride_s;
    O[out_base + d] = o_d / fmaxf(l, 1e-38f);

    // Save lse for backward (thread 0 only — lse is a scalar per query row).
    if (d == 0)
        lse[b * heads * seq + h * seq + qi] = m + __logf(fmaxf(l, 1e-38f));
}

// ---------------------------------------------------------------------------
// Backward kernel
//
// Grid : (batch, heads, seq)  — one block per query row
// Block: (d_head)
//
// dK and dV are accumulated across query rows via atomicAdd (zeroed before launch).
// dQ has no conflicts and is written directly.
//
// lse is read from the value saved in the forward pass — no recomputation needed.
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
    const int b  = blockIdx.x;
    const int h  = blockIdx.y;
    const int qi = blockIdx.z;
    const int d  = threadIdx.x;

    // Active-lane mask for warp ops (d_head <= 32 enforced by caller).
    // NOTE: (1u << 32) is UB; the ternary special-cases d_head == 32 to avoid it.
    const unsigned int mask = (d_head == 32) ? 0xffffffffu : (1u << d_head) - 1u;

    const int stride_b = heads * seq * d_head;
    const int stride_h =         seq * d_head;
    const int stride_s =               d_head;
    const int qi_base  = b * stride_b + h * stride_h + qi * stride_s;

    const float q_d  = __ldg(&Q [qi_base + d]);
    const float o_d  = __ldg(&O [qi_base + d]);
    const float do_d = __ldg(&dO[qi_base + d]);

    // D_i = dot(O_i, dO_i) — scalar used in the softmax gradient formula
    const float D = warp_dot(o_d, do_d, mask);

    // lse_i saved from forward — no need to re-scan K.
    const float lse_i = __ldg(&lse[b * heads * seq + h * seq + qi]);

    const int kv_end = causal ? qi + 1 : seq;

    float dq_d = 0.0f;
    for (int kv = 0; kv < kv_end; kv++) {
        const int   base  = b * stride_b + h * stride_h + kv * stride_s;
        const float k_d   = __ldg(&K[base + d]);
        const float v_d   = __ldg(&V[base + d]);

        const float score = warp_dot(q_d, k_d, mask) * scale;
        const float p     = __expf(score - lse_i);   // softmax weight p_ij

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
    flash_attn_bwd_kernel<<<grid, block>>>(
        q, k, v, o, do_, lse, dq, dk, dv, seq, heads, d_head, scale, causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "flash_attn_bwd kernel error: %s\n", cudaGetErrorString(err));
}

} // extern "C"
