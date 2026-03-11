fn main() {
    #[cfg(not(target_os = "macos"))]
    {
        let has_flash_attn = std::env::var("CARGO_FEATURE_FLASH_ATTN").is_ok();
        if !has_flash_attn {
            compile_cuda_kernels();
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn detect_sm() -> String {
    // Ask nvidia-smi for the compute capability of the first GPU (e.g. "7.5" -> "sm_75").
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();
    if let Ok(o) = out {
        if o.status.success() {
            if let Some(line) = String::from_utf8_lossy(&o.stdout).lines().next() {
                let cap = line.trim().replace('.', "");
                if !cap.is_empty() {
                    return format!("sm_{}", cap);
                }
            }
        }
    }
    println!(
        "cargo:warning=Could not detect GPU compute capability via nvidia-smi, defaulting to sm_75"
    );
    "sm_75".to_string()
}

#[cfg(not(target_os = "macos"))]
fn compile_cuda_kernels() {
    println!("cargo:rerun-if-changed=src/cuda_kernels/flash_attn.cu");

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let cuda_home = std::env::var("CUDA_HOME")
        .or_else(|_| std::env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    let arch = detect_sm();
    let arch_flag = format!("-arch={}", arch);
    println!("cargo:warning=Compiling flash_attn kernel for {}", arch);

    // Compile to object file
    let obj = format!("{}/flash_attn_kernel.o", out_dir);
    let status = std::process::Command::new("nvcc")
        .args([
            &arch_flag,
            "-O3",
            "--compiler-options",
            "-fPIC",
            "-c",
            "src/cuda_kernels/flash_attn.cu",
            "-o",
            &obj,
        ])
        .status()
        .expect("nvcc not found");
    assert!(status.success(), "nvcc compilation failed");

    // Archive into a static library — baked into the binary, no .so to find at runtime
    let lib = format!("{}/libflash_attn_kernel.a", out_dir);
    let status = std::process::Command::new("ar")
        .args(["rcs", &lib, &obj])
        .status()
        .expect("ar not found");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir);
    println!("cargo:rustc-link-lib=static=flash_attn_kernel");
    println!("cargo:rustc-link-search=native={}/lib64", cuda_home);
    println!("cargo:rustc-link-lib=dylib=cudart");
}
