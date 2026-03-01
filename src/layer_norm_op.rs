// Rust wrapper around the fused layer_norm CUDA kernels.

use candle_core::{Layout, Result, Shape, Tensor};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------
extern "C" {
    fn layer_norm_fwd(
        x: *const f32, w: *const f32, b: *const f32,
        out: *mut f32, mean_out: *mut f32, rstd_out: *mut f32,
        rows: i32, emb: i32, block_size: i32, eps: f32,
    );

    fn layer_norm_bwd(
        dy: *const f32, x: *const f32, w: *const f32,
        mean: *const f32, rstd: *const f32,
        dx: *mut f32, dw: *mut f32, db: *mut f32,
        rows: i32, emb: i32, block_size: i32,
    );
}

// ---------------------------------------------------------------------------
// Helpers (same pattern as flash_attn_op.rs)
// ---------------------------------------------------------------------------
type SharedSlice = Arc<Mutex<Option<candle_core::cuda_backend::cudarc::driver::CudaSlice<f32>>>>;

struct InjectCudaStorage {
    slice: SharedSlice,
    dev:   candle_core::cuda_backend::CudaDevice,
    shape: Shape,
}

impl candle_core::CustomOp1 for InjectCudaStorage {
    fn name(&self) -> &'static str { "inject_cuda_storage" }

    fn cpu_fwd(&self, _: &candle_core::CpuStorage, _: &Layout)
        -> Result<(candle_core::CpuStorage, Shape)>
    {
        candle_core::bail!("inject_cuda_storage is CUDA-only")
    }

    fn cuda_fwd(&self, _: &candle_core::CudaStorage, _: &Layout)
        -> Result<(candle_core::CudaStorage, Shape)>
    {
        use candle_core::cuda_backend::CudaStorageSlice;
        let slice = self.slice.lock().unwrap().take()
            .ok_or_else(|| candle_core::Error::msg("inject_cuda_storage: already consumed"))?;
        Ok((candle_core::CudaStorage {
            slice:  CudaStorageSlice::F32(slice),
            device: self.dev.clone(),
        }, self.shape.clone()))
    }
    // bwd returns Ok(None) by default — gradients don't need gradients
}

fn raw_cuda_ptr(t: &Tensor) -> Result<(*const f32, candle_core::cuda_backend::CudaDevice)> {
    use candle_core::{Storage, cuda_backend::cudarc::driver::DevicePtr};
    let (s, _) = t.storage_and_layout();
    match &*s {
        Storage::Cuda(c) => {
            let ptr = *c.as_cuda_slice::<f32>()?.device_ptr() as *const f32;
            let dev = c.device.clone();
            Ok((ptr, dev))
        }
        _ => candle_core::bail!("layer_norm: expected CUDA tensor"),
    }
}

// ---------------------------------------------------------------------------
// Main op
// ---------------------------------------------------------------------------
pub struct LayerNormOp {
    pub eps: f32,
    pub block_size: i32,
    // mean [rows] and rstd [rows] computed in cuda_fwd and consumed in bwd.
    pub mean_cache: SharedSlice,
    pub rstd_cache: SharedSlice,
}

impl candle_core::CustomOp3 for LayerNormOp {
    fn name(&self) -> &'static str { "layer_norm" }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage, _: &Layout,
        _: &candle_core::CpuStorage, _: &Layout,
        _: &candle_core::CpuStorage, _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        candle_core::bail!("layer_norm fused op: CPU not supported — should not be reached")
    }

    fn cuda_fwd(
        &self,
        x_st: &candle_core::CudaStorage, xl: &Layout,
        w_st: &candle_core::CudaStorage, _wl: &Layout,
        b_st: &candle_core::CudaStorage, _bl: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let dims = xl.shape().dims();
        let emb  = *dims.last().unwrap();
        let rows: usize = dims[..dims.len() - 1].iter().product();

        let n = rows * emb;

        let x_ptr = *x_st.as_cuda_slice::<f32>()?.device_ptr() as *const f32;
        let w_ptr = *w_st.as_cuda_slice::<f32>()?.device_ptr() as *const f32;
        let b_ptr = *b_st.as_cuda_slice::<f32>()?.device_ptr() as *const f32;

        let dev = x_st.device.clone();
        let out_slice  = unsafe { dev.alloc::<f32>(n)    }.w()?;
        let mean_slice = unsafe { dev.alloc::<f32>(rows) }.w()?;
        let rstd_slice = unsafe { dev.alloc::<f32>(rows) }.w()?;

        let out_ptr  = *out_slice.device_ptr()  as *mut f32;
        let mean_ptr = *mean_slice.device_ptr() as *mut f32;
        let rstd_ptr = *rstd_slice.device_ptr() as *mut f32;

        unsafe {
            layer_norm_fwd(
                x_ptr, w_ptr, b_ptr,
                out_ptr, mean_ptr, rstd_ptr,
                rows as i32, emb as i32, self.block_size, self.eps,
            );
        }

        *self.mean_cache.lock().unwrap() = Some(mean_slice);
        *self.rstd_cache.lock().unwrap() = Some(rstd_slice);

        let out_st = candle_core::CudaStorage {
            slice:  CudaStorageSlice::F32(out_slice),
            device: dev,
        };
        Ok((out_st, xl.shape().clone()))
    }

    fn bwd(
        &self,
        x: &Tensor, w: &Tensor, b: &Tensor,
        _out: &Tensor, dy: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>, Option<Tensor>)> {
        use candle_core::cuda_backend::WrapErr;
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;

        let dims = x.dims();
        let emb  = *dims.last().unwrap();
        let rows: usize = dims[..dims.len() - 1].iter().product();
        let n = rows * emb;

        let mean_slice = self.mean_cache.lock().unwrap().take()
            .ok_or_else(|| candle_core::Error::msg("layer_norm bwd: mean not in cache — was forward called first?"))?;
        let rstd_slice = self.rstd_cache.lock().unwrap().take()
            .ok_or_else(|| candle_core::Error::msg("layer_norm bwd: rstd not in cache — was forward called first?"))?;

        let mean_ptr = *mean_slice.device_ptr() as *const f32;
        let rstd_ptr = *rstd_slice.device_ptr() as *const f32;

        let (x_ptr,  dev) = raw_cuda_ptr(x)?;
        let (w_ptr,  _)   = raw_cuda_ptr(w)?;
        let (dy_ptr, _)   = raw_cuda_ptr(dy)?;

        let dx_slice = unsafe { dev.alloc::<f32>(n)   }.w()?;
        let dw_slice = unsafe { dev.alloc::<f32>(emb) }.w()?;
        let db_slice = unsafe { dev.alloc::<f32>(emb) }.w()?;

        let dx_ptr = *dx_slice.device_ptr() as *mut f32;
        let dw_ptr = *dw_slice.device_ptr() as *mut f32;
        let db_ptr = *db_slice.device_ptr() as *mut f32;

        unsafe {
            layer_norm_bwd(
                dy_ptr, x_ptr, w_ptr,
                mean_ptr, rstd_ptr,
                dx_ptr, dw_ptr, db_ptr,
                rows as i32, emb as i32, self.block_size,
            );
        }

        let inject = |slice, input: &Tensor, shape: Shape| -> Result<Tensor> {
            input.apply_op1(InjectCudaStorage {
                slice: Arc::new(Mutex::new(Some(slice))),
                dev:   dev.clone(),
                shape,
            })
        };

        let dx = inject(dx_slice, x, x.shape().clone())?;
        let dw = inject(dw_slice, w, w.shape().clone())?;
        let db = inject(db_slice, b, b.shape().clone())?;

        Ok((Some(dx), Some(dw), Some(db)))
    }
}

// ---------------------------------------------------------------------------
// Tests
//
// Both fused and reference run on the same CUDA device so FP ordering
// differences are small. Tests are skipped if no CUDA device is present.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var, D};

    fn cuda_device() -> Option<Device> {
        Device::new_cuda(0).ok()
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }

    // Reference: candle's decomposed ops (fast_sum + badd + etc.) on the same device.
    fn layer_norm_ref(
        x: &Tensor, w: &Tensor, b: &Tensor, eps: f64,
    ) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let xc   = x.broadcast_sub(&mean)?;
        let var  = xc.sqr()?.mean_keepdim(D::Minus1)?;
        let std  = (var + eps)?.sqrt()?;
        let xn   = xc.broadcast_div(&std)?;
        xn.broadcast_mul(w)?.broadcast_add(b)
    }

    // Build a LayerNormOp and run apply_op3 directly (bypasses the
    // layer_norm.rs contiguous() wrapper so we test the op itself).
    fn run_fused(
        x: &Tensor, w: &Tensor, b: &Tensor, emb: usize,
    ) -> candle_core::Result<Tensor> {
        let block_size = emb.next_power_of_two().min(1024) as i32;
        x.apply_op3(w, b, LayerNormOp {
            eps: 1e-5,
            block_size,
            mean_cache: Arc::new(Mutex::new(None)),
            rstd_cache: Arc::new(Mutex::new(None)),
        })
    }

    /// Forward: fused output ≈ reference (decomposed candle ops), same CUDA device.
    #[test]
    fn test_fwd_matches_reference() {
        let Some(dev) = cuda_device() else { return };

        let rows = 16usize;
        let emb  = 8usize;

        // Deterministic input: values spaced across a range including negatives.
        let x_data: Vec<f32> = (0..rows * emb)
            .map(|i| (i as f32 - (rows * emb / 2) as f32) * 0.1)
            .collect();
        let w_data: Vec<f32> = (0..emb).map(|i| 1.0 + i as f32 * 0.05).collect();
        let b_data: Vec<f32> = (0..emb).map(|i| i as f32 * 0.01).collect();

        let x = Tensor::from_slice(&x_data, (rows, emb), &dev).unwrap();
        let w = Tensor::from_slice(&w_data, (emb,),       &dev).unwrap();
        let b = Tensor::from_slice(&b_data, (emb,),       &dev).unwrap();

        let fused = run_fused(&x, &w, &b, emb).unwrap();
        let refer = layer_norm_ref(&x, &w, &b, 1e-5).unwrap();

        let fused_v: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        let refer_v: Vec<f32> = refer.flatten_all().unwrap().to_vec1().unwrap();

        let err = max_abs_diff(&fused_v, &refer_v);
        assert!(err < 1e-5, "forward max abs error {err} ≥ 1e-5");
    }

    /// Forward with 3D input [batch, seq, emb] — verifies row counting is correct.
    #[test]
    fn test_fwd_3d_input() {
        let Some(dev) = cuda_device() else { return };

        let (batch, seq, emb) = (3usize, 4usize, 8usize);
        let n = batch * seq * emb;
        let x_data: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.1).collect();
        let w_data: Vec<f32> = vec![1.0; emb];
        let b_data: Vec<f32> = vec![0.0; emb];

        let x = Tensor::from_slice(&x_data, (batch, seq, emb), &dev).unwrap();
        let w = Tensor::from_slice(&w_data, (emb,),             &dev).unwrap();
        let b = Tensor::from_slice(&b_data, (emb,),             &dev).unwrap();

        let fused = run_fused(&x, &w, &b, emb).unwrap();
        let refer = layer_norm_ref(&x, &w, &b, 1e-5).unwrap();

        let fused_v: Vec<f32> = fused.flatten_all().unwrap().to_vec1().unwrap();
        let refer_v: Vec<f32> = refer.flatten_all().unwrap().to_vec1().unwrap();

        let err = max_abs_diff(&fused_v, &refer_v);
        assert!(err < 1e-5, "3D forward max abs error {err} ≥ 1e-5");
    }

    /// Backward: dx, dw, db from fused op ≈ candle autograd through reference ops.
    #[test]
    fn test_bwd_matches_reference() {
        let Some(dev) = cuda_device() else { return };

        let rows = 16usize;
        let emb  = 8usize;

        let x_data: Vec<f32> = (0..rows * emb)
            .map(|i| (i as f32 - (rows * emb / 2) as f32) * 0.1)
            .collect();
        let w_data: Vec<f32> = (0..emb).map(|i| 1.0 + i as f32 * 0.05).collect();
        let b_data: Vec<f32> = (0..emb).map(|i| i as f32 * 0.01).collect();

        // --- Fused path ---
        let xf = Var::from_slice(&x_data, (rows, emb), &dev).unwrap();
        let wf = Var::from_slice(&w_data, (emb,),       &dev).unwrap();
        let bf = Var::from_slice(&b_data, (emb,),       &dev).unwrap();

        let out_f  = run_fused(xf.as_tensor(), wf.as_tensor(), bf.as_tensor(), emb).unwrap();
        let grads_f = out_f.sum_all().unwrap().backward().unwrap();
        let dx_f: Vec<f32> = grads_f.get(&xf).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let dw_f: Vec<f32> = grads_f.get(&wf).unwrap().to_vec1().unwrap();
        let db_f: Vec<f32> = grads_f.get(&bf).unwrap().to_vec1().unwrap();

        // --- Reference path ---
        let xr = Var::from_slice(&x_data, (rows, emb), &dev).unwrap();
        let wr = Var::from_slice(&w_data, (emb,),       &dev).unwrap();
        let br = Var::from_slice(&b_data, (emb,),       &dev).unwrap();

        let out_r  = layer_norm_ref(xr.as_tensor(), wr.as_tensor(), br.as_tensor(), 1e-5).unwrap();
        let grads_r = out_r.sum_all().unwrap().backward().unwrap();
        let dx_r: Vec<f32> = grads_r.get(&xr).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let dw_r: Vec<f32> = grads_r.get(&wr).unwrap().to_vec1().unwrap();
        let db_r: Vec<f32> = grads_r.get(&br).unwrap().to_vec1().unwrap();

        let dx_err = max_abs_diff(&dx_f, &dx_r);
        let dw_err = max_abs_diff(&dw_f, &dw_r);
        let db_err = max_abs_diff(&db_f, &db_r);

        assert!(dx_err < 1e-4, "dx max abs error {dx_err} ≥ 1e-4");
        assert!(dw_err < 1e-4, "dw max abs error {dw_err} ≥ 1e-4");
        assert!(db_err < 1e-4, "db max abs error {db_err} ≥ 1e-4");
    }
}
