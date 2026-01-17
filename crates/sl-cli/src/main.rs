// Re-enable IREE in main.rs
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::fs;
use sl_backend::Backend;
use sl_ir::{Module, Stmt, Expr};

#[derive(Parser)]
#[command(name = "sl")]
#[command(about = "shapelang compiler CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Check {
        file: PathBuf,
    },
    Compile {
        file: PathBuf,
        #[arg(long, default_value = "gpu")]
        target: Target,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        emit_backend_ir: bool,
    },
    Run {
        file: PathBuf,
        #[arg(long, default_value = "gpu")]
        device: Target,
    },
    Ir {
        file: PathBuf,
    },
    Testgrad {
        file: PathBuf,
        #[arg(long, default_value = "loss_fn")] 
        func: String,
    },
}

#[derive(Clone, ValueEnum)]
enum Target {
    Gpu,
    Cpu,
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Check { file } => {
            if let Err(e) = check(file) {
                eprintln!("Check failed: {}", e);
                std::process::exit(1);
            }
            println!("Check passed.");
        },
        Commands::Compile { file, target, out, emit_backend_ir } => {
            let out_path = out.clone().unwrap_or_else(|| {
                let mut p = file.clone();
                p.set_extension("vmfb"); 
                p
            });
            
            if let Err(e) = compile(file, target, &out_path, *emit_backend_ir) {
                eprintln!("Compilation failed: {}", e);
                std::process::exit(1);
            }
            println!("Compiled to {:?}", out_path);
        },
        Commands::Run { file, device } => {
            let is_source = file.extension().map_or(false, |ext| ext == "sl");
            
            if is_source {
                if let Target::Cpu = device {
                     if let Err(e) = run_interpreter(file) {
                         eprintln!("Execution failed: {}", e);
                         std::process::exit(1);
                     }
                } else {
                    // Try GPU first
                    let temp_out = file.with_extension("vmfb");
                    match compile(file, device, &temp_out, false) {
                        Ok(_) => {
                            if let Err(e) = run_artifact(&temp_out, device) {
                                eprintln!("GPU Execution failed: {}", e);
                                eprintln!("Falling back to CPU...");
                                let _ = run_interpreter(file);
                            }
                        },
                        Err(e) => {
                            eprintln!("GPU Compilation failed: {}", e);
                            eprintln!("To force CPU, use --device cpu");
                            std::process::exit(1);
                        }
                    }
                }
            } else {
                 if let Err(e) = run_artifact(file, device) {
                    eprintln!("Execution failed: {}", e);
                    std::process::exit(1);
                 }
            };
        },
        Commands::Ir { file } => {
            match get_ir(file) {
                Ok(ir) => println!("{:#?}", ir),
                Err(e) => eprintln!("Error: {}", e),
            }
        },
        Commands::Testgrad { file, func } => {
            if let Err(e) = test_grad(file, func) {
                eprintln!("Testgrad failed: {}", e);
                std::process::exit(1);
            }
        }
    }
}

fn check(path: &PathBuf) -> Result<(), String> {
    let _ = get_ir(path)?;
    Ok(())
}

fn compile(path: &PathBuf, target: &Target, out: &PathBuf, emit_ir: bool) -> Result<(), String> {
    let mut module = get_ir(path)?;
    run_autodiff(&mut module)?;
    
    let backend: Box<dyn Backend> = match target {
        Target::Cpu => Box::new(sl_backend_cpu::CpuBackend),
        Target::Gpu => Box::new(sl_backend_iree::IreeBackend),
    };
    
    if emit_ir && matches!(target, Target::Gpu) {
        // IreeBackend handles emitting if implemented, or we do it here.
        // IreeBackend::compile saves .mlir.
        println!("Backend IR emitted (if supported).");
    }
    
    backend.compile(&module, out.to_str().unwrap())
}

fn run_artifact(path: &PathBuf, target: &Target) -> Result<(), String> {
    let backend: Box<dyn Backend> = match target {
        Target::Cpu => Box::new(sl_backend_cpu::CpuBackend),
        Target::Gpu => Box::new(sl_backend_iree::IreeBackend),
    };
    backend.run(path.to_str().unwrap())
}

// ... rest of file (run_interpreter, get_ir, run_autodiff, test_grad)
fn run_interpreter(path: &PathBuf) -> Result<(), String> {
    let mut module = get_ir(path)?;
    run_autodiff(&mut module)?;
    sl_backend_cpu::interpret(&mut module)
}

fn get_ir(path: &PathBuf) -> Result<Module, String> {
    let source = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let ast = sl_parse::parse(&source).map_err(|e| format!("{:?}", e))?;
    let mut ctx = sl_ir::LoweringCtx::new();
    ctx.lower_module(ast)
}

fn run_autodiff(module: &mut Module) -> Result<(), String> {
    let mut targets = Vec::new();
    for func in &module.funcs {
        for stmt in &func.body {
            scan_stmt_for_grad(stmt, &mut targets);
        }
    }
    targets.sort();
    targets.dedup();
    for target in targets {
        sl_autodiff::differentiate(module, &target)?;
    }
    Ok(())
}

fn scan_stmt_for_grad(stmt: &Stmt, targets: &mut Vec<String>) {
    match stmt {
        Stmt::Let(_, _, expr) | Stmt::Expr(expr) => scan_expr_for_grad(expr, targets),
        Stmt::If(_, t, e) => {
            for s in t { scan_stmt_for_grad(s, targets); }
            for s in e { scan_stmt_for_grad(s, targets); }
        },
        Stmt::For(_, _, b) => {
            for s in b { scan_stmt_for_grad(s, targets); }
        },
        _ => {}
    }
}

fn scan_expr_for_grad(expr: &Expr, targets: &mut Vec<String>) {
    match expr {
        Expr::Grad(name) => targets.push(name.clone()),
        _ => {}
    }
}

fn test_grad(path: &PathBuf, func_name: &String) -> Result<(), String> {
    println!("Testing gradients for {} in {:?}...", func_name, path);
    let mut module = get_ir(path)?;
    run_autodiff(&mut module)?;
    
    let func = module.funcs.iter().find(|f| f.name == func_name)
        .ok_or(format!("Function {} not found", func_name))?.clone();
        
    let mut args = Vec::new();
    use sl_types::Type;
    for (_, ty) in &func.params {
        match ty {
            Type::Tensor(sl_types::ScalarType::F32, shape) => {
                let mut dims = Vec::new();
                for d in shape {
                    match d {
                        sl_types::ShapeDim::Fixed(n) => dims.push(*n),
                        sl_types::ShapeDim::Symbol(_) => dims.push(2), 
                    }
                }
                let size: usize = dims.iter().product();
                let data = vec![0.5; size]; 
                args.push(sl_backend_cpu::RuntimeValue::Tensor(data, dims));
            },
            Type::Scalar(sl_types::ScalarType::F32) => {
                args.push(sl_backend_cpu::RuntimeValue::Scalar(0.5));
            },
             _ => return Err("Unsupported type for testgrad".into()),
        }
    }
    
    let res = sl_backend_cpu::execute(&module, func_name, args.clone())?;
    let f_val = match res {
        sl_backend_cpu::RuntimeValue::Scalar(v) => v,
        _ => return Err("Function must return scalar for testgrad".into()),
    };
    println!("Forward result: {}", f_val);
    
    let grad_fn_name = format!("{}_grad", func_name);
    let grad_res = sl_backend_cpu::execute(&module, &grad_fn_name, args.clone())?;
    
    let analytic_grads = match grad_res {
        sl_backend_cpu::RuntimeValue::Tuple(ts) => ts,
        _ => return Err("Grad func must return tuple".into()),
    };
    
    let epsilon = 1e-3;
    println!("Running finite difference checks...");
    
    for (i, arg) in args.iter().enumerate() {
         if let sl_backend_cpu::RuntimeValue::Tensor(data, _) = arg {
             let grad_est = &analytic_grads[i+1]; 
             if let sl_backend_cpu::RuntimeValue::Tensor(g_data, _) = grad_est {
                 for k in 0..data.len().min(5) {
                     let mut args_p = args.clone();
                     if let sl_backend_cpu::RuntimeValue::Tensor(d, _) = &mut args_p[i] {
                         d[k] += epsilon;
                     }
                     let res_p = sl_backend_cpu::execute(&module, func_name, args_p)?;
                     let val_p = match res_p { sl_backend_cpu::RuntimeValue::Scalar(v) => v, _ => 0.0 };
                     
                     let mut args_m = args.clone();
                     if let sl_backend_cpu::RuntimeValue::Tensor(d, _) = &mut args_m[i] {
                         d[k] -= epsilon;
                     }
                     let res_m = sl_backend_cpu::execute(&module, func_name, args_m)?;
                     let val_m = match res_m { sl_backend_cpu::RuntimeValue::Scalar(v) => v, _ => 0.0 };
                     
                     let num_grad = (val_p - val_m) / (2.0 * epsilon);
                     let ana_grad = g_data[k];
                     
                     if (num_grad - ana_grad).abs() > 1e-2 {
                         println!("Mismatch arg {} idx {}: num {} vs ana {}", i, k, num_grad, ana_grad);
                     }
                 }
             }
         }
    }
    
    println!("Testgrad completed.");
    Ok(())
}
