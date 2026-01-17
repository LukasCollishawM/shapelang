use sl_backend::Backend;
use sl_ir::{Module, Function, Stmt, Expr, Value, Literal, BinaryOp, UnaryOp};
use std::collections::HashMap;
// use rand::Rng; // Removed

pub struct CpuBackend;

impl Backend for CpuBackend {
    fn name(&self) -> &str { "cpu" }
    
    fn compile(&self, _module: &Module, _out_path: &str) -> Result<(), String> {
        std::fs::write(_out_path, "cpu-artifact").map_err(|e| e.to_string())
    }

    fn run(&self, artifact_path: &str) -> Result<(), String> {
        println!("Running on CPU...");
        Ok(())
    }
}

pub fn execute(module: &Module, func_name: &str, args: Vec<RuntimeValue>) -> Result<RuntimeValue, String> {
    eval_func_call(module, func_name, args)
}

pub fn interpret(module: &mut Module) -> Result<(), String> {
    let main = module.funcs.iter().find(|f| f.name == "main")
        .ok_or("No main function")?;
    
    let mut env = HashMap::new();
    eval_function(module, main.name.clone(), &mut env)
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    Scalar(f32),
    Tensor(Vec<f32>, Vec<usize>),
    Tuple(Vec<RuntimeValue>),
    FuncRef(String),
    Unit,
}

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.state >> 32) as f32 / 4294967296.0
    }
}

use std::cell::RefCell;
thread_local! {
    static RNG: RefCell<Lcg> = RefCell::new(Lcg::new(42));
}

fn next_random() -> f32 {
    RNG.with(|rng| rng.borrow_mut().next_f32())
}

fn eval_function(module: &Module, func_name: String, env: &mut HashMap<String, RuntimeValue>) -> Result<(), String> {
    let func = module.funcs.iter().find(|f| f.name == func_name)
        .ok_or(format!("Function {} not found", func_name))?;
        
    for stmt in &func.body {
        match stmt {
            Stmt::Let(name, _, expr) => {
                let val = eval_expr(module, expr, env)?;
                env.insert(name.clone(), val);
            },
            Stmt::Expr(e) => {
                eval_expr(module, e, env)?;
            },
            Stmt::Return(v) => {
                return Err("Return not handled in void eval".to_string());
            },
             Stmt::For(var, n, body) => {
                 for i in 0..*n {
                     env.insert(var.clone(), RuntimeValue::Scalar(i as f32));
                     eval_block(module, body, env)?;
                 }
            },
            Stmt::If(cond, then_block, else_block) => {
                let c = resolve_value(cond, env)?;
                let b = match c {
                    RuntimeValue::Scalar(v) => v != 0.0,
                    _ => false,
                };
                if b {
                    eval_block(module, then_block, env)?;
                } else {
                    eval_block(module, else_block, env)?;
                }
            }
        }
    }
    Ok(())
}

fn eval_block(module: &Module, stmts: &[Stmt], env: &mut HashMap<String, RuntimeValue>) -> Result<Option<RuntimeValue>, String> {
    for stmt in stmts {
        match stmt {
             Stmt::Let(name, _, expr) => {
                let val = eval_expr(module, expr, env)?;
                env.insert(name.clone(), val);
            },
            Stmt::Expr(e) => {
                eval_expr(module, e, env)?;
            },
            Stmt::Return(v) => {
                let val = resolve_value(v, env)?;
                return Ok(Some(val));
            },
             _ => { /* For/If recurse */ }
        }
    }
    Ok(None)
}

fn resolve_value(v: &Value, env: &HashMap<String, RuntimeValue>) -> Result<RuntimeValue, String> {
    match v {
        Value::Var(n) => env.get(n).cloned().ok_or(format!("Var {} not found", n)),
        Value::Literal(l) => match l {
            Literal::Int(i) => Ok(RuntimeValue::Scalar(*i as f32)),
            Literal::Float(f) => Ok(RuntimeValue::Scalar(*f as f32)),
            Literal::Bool(b) => Ok(RuntimeValue::Scalar(if *b { 1.0 } else { 0.0 })),
        },
        Value::String(_) => Ok(RuntimeValue::Scalar(0.0)),
    }
}

fn eval_expr(module: &Module, expr: &Expr, env: &mut HashMap<String, RuntimeValue>) -> Result<RuntimeValue, String> {
    match expr {
        Expr::Literal(l) => resolve_value(&Value::Literal(l.clone()), env),
        Expr::Binary(op, lhs, rhs) => {
            let l = resolve_value(lhs, env)?;
            let r = resolve_value(rhs, env)?;
            match (l, r) {
                (RuntimeValue::Scalar(a), RuntimeValue::Scalar(b)) => match op {
                    BinaryOp::Add => Ok(RuntimeValue::Scalar(a + b)),
                    BinaryOp::Sub => Ok(RuntimeValue::Scalar(a - b)),
                    BinaryOp::Mul => Ok(RuntimeValue::Scalar(a * b)),
                    BinaryOp::Div => Ok(RuntimeValue::Scalar(a / b)),
                },
                (RuntimeValue::Tensor(d1, s1), RuntimeValue::Tensor(d2, s2)) => {
                    if s1 == s2 {
                        let res_data: Vec<f32> = d1.iter().zip(d2.iter()).map(|(a, b)| match op {
                            BinaryOp::Add => a + b,
                            BinaryOp::Sub => a - b,
                            BinaryOp::Mul => a * b,
                            BinaryOp::Div => a / b,
                        }).collect();
                        Ok(RuntimeValue::Tensor(res_data, s1))
                    } else if s1.len() == 2 && s2.len() == 2 && s1[1] == s2[1] && s2[0] == 1 {
                        let m = s1[0];
                        let n = s1[1];
                        let mut res_data = Vec::with_capacity(m * n);
                        for i in 0..m {
                            for j in 0..n {
                                let a = d1[i * n + j];
                                let b = d2[j];
                                res_data.push(match op {
                                    BinaryOp::Add => a + b,
                                    BinaryOp::Sub => a - b,
                                    BinaryOp::Mul => a * b,
                                    BinaryOp::Div => a / b,
                                });
                            }
                        }
                        Ok(RuntimeValue::Tensor(res_data, s1))
                    } else {
                        Err(format!("Shape mismatch {:?} {:?}", s1, s2))
                    }
                },
                (RuntimeValue::Scalar(s), RuntimeValue::Tensor(d, sh)) | (RuntimeValue::Tensor(d, sh), RuntimeValue::Scalar(s)) => {
                    let res_data: Vec<f32> = d.iter().map(|a| match op {
                         BinaryOp::Add => a + s,
                         BinaryOp::Sub => a - s, 
                         BinaryOp::Mul => a * s,
                         BinaryOp::Div => a / s,
                    }).collect();
                     Ok(RuntimeValue::Tensor(res_data, sh))
                },
                _ => Err("Binary op type mismatch".into())
            }
        },
        Expr::Call(name, args) => {
             let mut r_args = Vec::new();
             for arg in args {
                 r_args.push(resolve_value(arg, env)?);
             }
             
             match name.as_str() {
                 "print" => {
                     println!("{:?}", r_args[0]);
                     Ok(RuntimeValue::Unit)
                 },
                 "zeros" | "ones" | "randn" => {
                     let mut shape = Vec::new();
                     for a in r_args {
                         if let RuntimeValue::Scalar(v) = a {
                             shape.push(v as usize);
                         }
                     }
                     let size = shape.iter().product();
                     let data = if name == "zeros" {
                         vec![0.0; size]
                     } else if name == "ones" {
                         vec![1.0; size]
                     } else {
                         (0..size).map(|_| next_random() - 0.5).collect()
                     };
                     Ok(RuntimeValue::Tensor(data, shape))
                 },
                 "transpose" => {
                     if let RuntimeValue::Tensor(d, s) = &r_args[0] {
                         if s.len() == 2 {
                             let rows = s[0];
                             let cols = s[1];
                             let mut new_data = vec![0.0; rows * cols];
                             for r in 0..rows {
                                 for c in 0..cols {
                                     new_data[c * rows + r] = d[r * cols + c];
                                 }
                             }
                             Ok(RuntimeValue::Tensor(new_data, vec![cols, rows]))
                         } else {
                             Err("Transpose only 2D".into())
                         }
                     } else {
                         Err("Transpose expects tensor".into())
                     }
                 },
                 "tuple_get" => {
                     if let RuntimeValue::Tuple(ts) = &r_args[0] {
                         if let RuntimeValue::Scalar(idx) = r_args[1] {
                             let i = idx as usize;
                             if i < ts.len() {
                                 Ok(ts[i].clone())
                             } else {
                                 Err("Tuple index out of bounds".into())
                             }
                         } else {
                             Err("Tuple index must be scalar".into())
                         }
                     } else {
                         Err("tuple_get expects tuple".into())
                     }
                 },
                 "cross_entropy_logits" => {
                     if let (RuntimeValue::Tensor(logits, s), RuntimeValue::Tensor(labels, _)) = (&r_args[0], &r_args[1]) {
                         let rows = s[0];
                         let cols = s[1];
                         let mut total_loss = 0.0;
                         
                         for r in 0..rows {
                             let start = r * cols;
                             let end = start + cols;
                             let logit_row = &logits[start..end];
                             let label_row = &labels[start..end];
                             
                             let max_val = logit_row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                             let mut sum_exp = 0.0;
                             let mut exps = vec![0.0; cols];
                             for i in 0..cols {
                                 exps[i] = (logit_row[i] - max_val).exp();
                                 sum_exp += exps[i];
                             }
                             
                             let mut row_loss = 0.0;
                             for i in 0..cols {
                                 let log_prob = exps[i].ln() - sum_exp.ln();
                                 row_loss -= label_row[i] * log_prob;
                             }
                             total_loss += row_loss;
                         }
                         Ok(RuntimeValue::Scalar(total_loss / rows as f32))
                     } else {
                         Err("CE expects tensors".into())
                     }
                 },
                 "softmax" => {
                      if let RuntimeValue::Tensor(d, s) = &r_args[0] {
                           let mut out = d.clone();
                           let rows = s[0];
                           let cols = s[1];
                           for r in 0..rows {
                               let start = r * cols;
                               let end = start + cols;
                               let slice = &d[start..end];
                               let max_val = slice.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                               let mut sum = 0.0;
                               for i in 0..cols {
                                   let v = (slice[i] - max_val).exp();
                                   out[start + i] = v;
                                   sum += v;
                               }
                               for i in 0..cols {
                                   out[start + i] /= sum;
                               }
                           }
                           Ok(RuntimeValue::Tensor(out, s.clone()))
                      } else {
                          Err("Softmax expects tensor".into())
                      }
                 },
                 "onehot" => {
                      if let (RuntimeValue::Tensor(d, s), RuntimeValue::Scalar(c)) = (&r_args[0], &r_args[1]) {
                          let rows = s[0];
                          let classes = *c as usize;
                          let mut out = vec![0.0; rows * classes];
                          for r in 0..rows {
                              let idx = d[r].round() as usize; 
                              if idx < classes {
                                  out[r * classes + idx] = 1.0;
                              }
                          }
                          Ok(RuntimeValue::Tensor(out, vec![rows, classes]))
                      } else {
                          Err("onehot expects tensor and scalar".into())
                      }
                 },
                 "mean" => {
                     if let RuntimeValue::Tensor(d, s) = &r_args[0] {
                         let sum: f32 = d.iter().sum();
                         let count = s.iter().product::<usize>() as f32;
                         Ok(RuntimeValue::Scalar(sum / count))
                     } else {
                         Err("mean expects tensor".into())
                     }
                 },
                 "ones_like" | "zeros_like" => {
                     if let RuntimeValue::Tensor(_, s) = &r_args[0] {
                         let size = s.iter().product();
                         let data = if name == "ones_like" {
                             vec![1.0; size]
                         } else {
                             vec![0.0; size]
                         };
                         Ok(RuntimeValue::Tensor(data, s.clone()))
                     } else {
                         Err("ones_like/zeros_like expects tensor".into())
                     }
                 },
                 _ => {
                     if let Some(f_ref) = env.get(name) {
                         if let RuntimeValue::FuncRef(real_name) = f_ref {
                             return eval_func_call(module, real_name, r_args);
                         }
                     }
                     eval_func_call(module, name, r_args)
                 }
             }
        },
        Expr::MatMul(lhs, rhs) => {
            let l = resolve_value(lhs, env)?;
            let r = resolve_value(rhs, env)?;
            if let (RuntimeValue::Tensor(d1, s1), RuntimeValue::Tensor(d2, s2)) = (l, r) {
                let m = s1[0];
                let k = s1[1];
                let n = s2[1];
                if s1[1] != s2[0] { return Err("Matmul dim mismatch".into()); }
                let mut res = vec![0.0; m * n];
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for x in 0..k {
                            sum += d1[i * k + x] * d2[x * n + j];
                        }
                        res[i * n + j] = sum;
                    }
                }
                Ok(RuntimeValue::Tensor(res, vec![m, n]))
            } else {
                Err("Matmul expects tensors".into())
            }
        },
        Expr::Grad(fn_name) => {
             Ok(RuntimeValue::FuncRef(format!("{}_grad", fn_name)))
        },
        Expr::Tuple(vals) => {
            let mut res = Vec::new();
            for v in vals {
                res.push(resolve_value(v, env)?);
            }
            Ok(RuntimeValue::Tuple(res))
        },
        Expr::Unary(op, val) => {
            let v = resolve_value(val, env)?;
            match v {
                RuntimeValue::Scalar(x) => match op {
                    UnaryOp::Neg => Ok(RuntimeValue::Scalar(-x)),
                    UnaryOp::Relu => Ok(RuntimeValue::Scalar(if x > 0.0 { x } else { 0.0 })),
                     _ => Err("Unary op not impl".into())
                },
                RuntimeValue::Tensor(d, s) => {
                    let res: Vec<f32> = d.iter().map(|&x| match op {
                        UnaryOp::Neg => -x,
                        UnaryOp::Relu => if x > 0.0 { x } else { 0.0 },
                         _ => x
                    }).collect();
                    Ok(RuntimeValue::Tensor(res, s))
                },
                _ => Err("Unary type mismatch".into())
            }
        },
        _ => Err("Expr not impl".into())
    }
}

fn eval_func_call(module: &Module, name: &str, args: Vec<RuntimeValue>) -> Result<RuntimeValue, String> {
    let func = module.funcs.iter().find(|f| f.name == name)
        .ok_or(format!("Function {} not found", name))?;
    
    let mut new_env = HashMap::new();
    for (i, (param_name, _)) in func.params.iter().enumerate() {
        if i < args.len() {
            new_env.insert(param_name.clone(), args[i].clone());
        }
    }
    
    if let Some(ret) = eval_block(module, &func.body, &mut new_env)? {
        Ok(ret)
    } else {
        Ok(RuntimeValue::Unit)
    }
}
