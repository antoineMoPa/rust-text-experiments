// Rust wrapper around our homebrew flash_attn CUDA kernels.

use candle_core::{Layout, Result, Shape, Tensor};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------
extern "C" {
    fn flash_attn_fwd(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
        lse: *mut f32,
        batch: i32,
        seq: i32,
        heads: i32,
        d_head: i32,
        scale: f32,
        causal: bool,
    );

    fn flash_attn_bwd(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *const f32,
        do_: *const f32,
        lse: *const f32,
        dq: *mut f32,
        dk: *mut f32,
        dv: *mut f32,
        batch: i32,
        seq: i32,
        heads: i32,
        d_head: i32,
        scale: f32,
        causal: bool,
    );
}

// ---------------------------------------------------------------------------
// Helper: wrap a pre-computed CudaSlice into a Tensor.
//
// apply_op1 is the only public API to produce a Tensor from a CudaStorage.
// We pass a dummy input tensor just to go through the CUDA dispatch; the op
// ignores it and returns our pre-allocated slice.
// ---------------------------------------------------------------------------
type SharedSlice = Arc<Mutex<Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>>>;

struct InjectCudaStorage {
    slice: SharedSlice,
    dev: candle_core::cuda_backend::CudaDevice,
    shape: Shape,
}

impl candle_core::CustomOp1 for InjectCudaStorage {
    fn name(&self) -> &'static str {
        "inject_cuda_storage"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("inject_cuda_storage is CUDA-only")
    }

    fn cuda_fwd(
        &self,
        _: &candle_core::CudaStorage,
        _: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::CudaStorageSlice;
        let slice = self
            .slice
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| candle_core::Error::msg("inject_cuda_storage: already consumed"))?;
        Ok((
            candle_core::CudaStorage {
                slice: CudaStorageSlice::F32(slice),
                device: self.dev.clone(),
            },
            self.shape.clone(),
        ))
    }
    // bwd defaults to Ok(None) — gradients don't need gradients
}

// Extract the raw CUDA device pointer from a Tensor (f32). Drops the storage
// lock immediately; the pointer stays valid as long as the Tensor lives.
fn raw_cuda_ptr(t: &Tensor) -> Result<(*const f32, candle_core::cuda_backend::CudaDevice)> {
    use candle_core::{cuda_backend::cudarc::driver::DevicePtr, Storage};
    let (s, _) = t.storage_and_layout();
    match &*s {
        Storage::Cuda(c) => {
            let ptr = *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32;
            let dev = c.device.clone();
            Ok((ptr, dev))
        }
        _ => candle_core::bail!("flash_attn bwd: expected CUDA tensor"),
    }
}

// ---------------------------------------------------------------------------
// Main op
// ---------------------------------------------------------------------------
pub struct FlashAttnOp {
    pub scale: f32,
    pub causal: bool,
    // lse [batch, heads, seq] computed in cuda_fwd and consumed in bwd.
    // Stored here because candle's bwd() signature has no other channel for saved tensors.
    lse_cache: SharedSlice,
}

impl candle_core::CustomOp3 for FlashAttnOp {
    fn name(&self) -> &'static str {
        "flash_attn"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("flash_attn: CPU not supported, use a CUDA device")
    }

    fn cuda_fwd(
        &self,
        q_st: &candle_core::CudaStorage,
        ql: &Layout,
        k_st: &candle_core::CudaStorage,
        kl: &Layout,
        v_st: &candle_core::CudaStorage,
        vl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        // Tensors arrive in [batch, heads, seq, d_head] order after the transpose in attention_block.
        let dims = ql.shape().dims();
        let (batch, heads, seq, d_head) = match dims {
            &[b, h, s, d] => (b, h, s, d),
            _ => candle_core::bail!(
                "flash_attn: expected [batch, heads, seq, d_head], got {:?}",
                dims
            ),
        };

        if d_head > 1024 {
            candle_core::bail!("flash_attn: d_head={} exceeds CUDA block size limit of 1024", d_head);
        }

        // Raw device pointers are base-of-allocation; verify no logical offset is hiding.
        if ql.start_offset() != 0 || kl.start_offset() != 0 || vl.start_offset() != 0 {
            candle_core::bail!("flash_attn: Q/K/V must have zero storage offset — call .contiguous() before flash_attn");
        }

        let n = batch * heads * seq * d_head;
        let lse_n = batch * heads * seq;

        let q_ptr = *q_st.as_cuda_slice::<f32>()?.device_ptr() as *const f32;
        let k_ptr = *k_st.as_cuda_slice::<f32>()?.device_ptr() as *const f32;
        let v_ptr = *v_st.as_cuda_slice::<f32>()?.device_ptr() as *const f32;

        let dev = q_st.device.clone();
        let o_slice = unsafe { dev.alloc::<f32>(n) }.w()?;
        let lse_slice = unsafe { dev.alloc::<f32>(lse_n) }.w()?;

        let o_ptr = *o_slice.device_ptr() as *mut f32;
        let lse_ptr = *lse_slice.device_ptr() as *mut f32;

        unsafe {
            flash_attn_fwd(
                q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                lse_ptr,
                batch as i32,
                seq as i32,
                heads as i32,
                d_head as i32,
                self.scale,
                self.causal,
            );
        }

        // Keep lse alive in the op; backward will consume it via lse_cache.
        *self.lse_cache.lock().unwrap() = Some(lse_slice);

        let out_st = candle_core::CudaStorage {
            slice: CudaStorageSlice::F32(o_slice),
            device: dev,
        };
        Ok((out_st, ql.shape().clone()))
    }

    fn bwd(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        o: &Tensor,
        do_: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        use candle_core::cuda_backend::WrapErr;

        // q, k, v are [batch, heads, seq, d_head] after the layout change.
        let (batch, heads, seq, d_head) = q.dims4()?;
        let n = batch * heads * seq * d_head;

        // Retrieve lse saved during forward (one-shot: take() clears the cache).
        let lse_slice = self.lse_cache.lock().unwrap().take().ok_or_else(|| {
            candle_core::Error::msg("flash_attn bwd: lse not in cache — was forward called first?")
        })?;
        let lse_ptr = *lse_slice.device_ptr() as *const f32;

        // Extract pointers (locks are held only for the duration of ptr extraction)
        let (q_ptr, dev) = raw_cuda_ptr(q)?;
        let (k_ptr, _) = raw_cuda_ptr(k)?;
        let (v_ptr, _) = raw_cuda_ptr(v)?;
        let (o_ptr, _) = raw_cuda_ptr(o)?;
        let (do_ptr, _) = raw_cuda_ptr(do_)?;

        // dq: each (b, h, qi) block owns a disjoint slice and writes directly — no zeroing needed.
        // dk, dv: multiple qi-blocks atomicAdd into the same kv positions — zeroed by the kernel.
        let dq_slice = unsafe { dev.alloc::<f32>(n) }.w()?;
        let dk_slice = unsafe { dev.alloc::<f32>(n) }.w()?;
        let dv_slice = unsafe { dev.alloc::<f32>(n) }.w()?;

        let dq_ptr = *dq_slice.device_ptr() as *mut f32;
        let dk_ptr = *dk_slice.device_ptr() as *mut f32;
        let dv_ptr = *dv_slice.device_ptr() as *mut f32;

        unsafe {
            flash_attn_bwd(
                q_ptr,
                k_ptr,
                v_ptr,
                o_ptr,
                do_ptr,
                lse_ptr,
                dq_ptr,
                dk_ptr,
                dv_ptr,
                batch as i32,
                seq as i32,
                heads as i32,
                d_head as i32,
                self.scale,
                self.causal,
            );
        }

        // Wrap raw CUDA slices into Tensors via the InjectCudaStorage helper
        let shape = q.shape().clone();
        let inject = |slice, input: &Tensor| -> Result<Tensor> {
            input.apply_op1(InjectCudaStorage {
                slice: Arc::new(Mutex::new(Some(slice))),
                dev: dev.clone(),
                shape: shape.clone(),
            })
        };

        let dq = inject(dq_slice, q)?;
        let dk = inject(dk_slice, k)?;
        let dv = inject(dv_slice, v)?;

        Ok((Some(dq), Some(dk), Some(dv)))
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// q, k, v: [batch, heads, seq, d_head] f32 on CUDA (heads-first layout).
/// Returns:  [batch, heads, seq, d_head] f32.
pub fn flash_attn(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    q.apply_op3(
        k,
        v,
        FlashAttnOp {
            scale,
            causal,
            lse_cache: Arc::new(Mutex::new(None)),
        },
    )
}
