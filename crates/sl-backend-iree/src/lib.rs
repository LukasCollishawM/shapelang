use sl_backend::Backend;
use sl_ir::{Module, Function, Stmt, Expr, Value, Literal, BinaryOp, UnaryOp};
use sl_types::{Type, ScalarType};
use std::process::Command;
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::Write;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub struct IreeBackend;

impl Backend for IreeBackend {
    fn name(&self) -> &str { "gpu" }

    fn compile(&self, module: &Module, out_path: &str) -> Result<(), String> {
        let mlir = generate_mlir(module);
        
        // Caching logic
        let hash = calculate_hash(&mlir);
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("shapelang").join("iree");
        fs::create_dir_all(&cache_dir).map_err(|e| e.to_string())?;
        
        let cached_vmfb = cache_dir.join(format!("{}.vmfb", hash));
        
        if cached_vmfb.exists() {
            println!("Using cached artifact: {:?}", cached_vmfb);
            fs::copy(&cached_vmfb, out_path).map_err(|e| e.to_string())?;
            return Ok(());
        }

        let mlir_path = format!("{}.mlir", out_path);
        let mut f = File::create(&mlir_path).map_err(|e| e.to_string())?;
        f.write_all(mlir.as_bytes()).map_err(|e| e.to_string())?;

        // Locate iree-compile
        let compiler = find_or_download_iree_compiler()?;
        
        // Compile to Vulkan (single target)
        println!("Compiling to Vulkan SPIR-V...");
        let output = Command::new(compiler)
            .arg("--iree-hal-target-backends=vulkan-spirv")
            .arg(&mlir_path)
            .arg("-o")
            .arg(out_path)
            .output()
            .map_err(|e| e.to_string())?;
            
        if !output.status.success() {
            return Err(format!("IREE compilation failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        
        // Cache result
        let _ = fs::copy(out_path, cached_vmfb);

        Ok(())
    }

    fn run(&self, artifact_path: &str) -> Result<(), String> {
        let runtime = find_or_download_iree_runtime()?;
        
        println!("Executing on GPU (IREE/Vulkan)...");
        let output = Command::new(runtime)
            .arg(format!("--module={}", artifact_path))
            .arg("--function=main")
            .output()
            .map_err(|e| e.to_string())?;
            
        if !output.status.success() {
             return Err(format!("IREE execution failed: {}", String::from_utf8_lossy(&output.stderr)));
        }
        
        println!("{}", String::from_utf8_lossy(&output.stdout));
        Ok(())
    }
}

fn generate_mlir(module: &Module) -> String {
    let mut out = String::new();
    out.push_str("module {\n");
    for func in &module.funcs {
        // Signature
        out.push_str(&format!("  func.func @{}(", func.name));
        out.push_str(") -> (");
        out.push_str(") {\n");
        out.push_str("    return\n");
        out.push_str("  }\n");
    }
    out.push_str("}\n");
    out
}

fn calculate_hash<T: Hash>(t: &T) -> u64 {
    let mut s = DefaultHasher::new();
    t.hash(&mut s);
    s.finish()
}

fn find_or_download_iree_compiler() -> Result<String, String> {
    if let Ok(path) = which::which("iree-compile") {
        return Ok(path.to_string_lossy().to_string());
    }
    Err("iree-compile not found in PATH. Please install IREE.".to_string())
}

fn find_or_download_iree_runtime() -> Result<String, String> {
    if let Ok(path) = which::which("iree-run-module") {
        return Ok(path.to_string_lossy().to_string());
    }
    Err("iree-run-module not found in PATH. Please install IREE.".to_string())
}
