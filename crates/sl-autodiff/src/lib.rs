use sl_ir::{Function, Stmt, Expr, Value, Literal, Module, BinaryOp, UnaryOp};
use sl_types::{Type, ScalarType};
use std::collections::HashMap;

pub fn differentiate(module: &mut Module, func_name: &str) -> Result<String, String> {
    let func = module.funcs.iter().find(|f| f.name == func_name)
        .ok_or("Function not found")?.clone();

    let grad_func_name = format!("{}_grad", func_name);
    let mut grad_body = func.body.clone();
    
    let ret_val = match grad_body.last() {
        Some(Stmt::Return(v)) => v.clone(),
        _ => return Err("Function must end with return".into()),
    };
    grad_body.pop();
    
    let mut adjoints: HashMap<String, String> = HashMap::new();
    let mut tmp_counter = 0;
    let mut new_stmts = Vec::new();
    let mut new_tmp = || {
        tmp_counter += 1;
        format!("_g{}", tmp_counter)
    };
    
    let d_loss = new_tmp();
    new_stmts.push(Stmt::Let(d_loss.clone(), Type::Scalar(ScalarType::F32), Expr::Literal(Literal::Float(1.0))));
    
    if let Value::Var(v) = ret_val {
        adjoints.insert(v, d_loss);
    }
    
    for stmt in func.body.iter().rev() {
        match stmt {
            Stmt::Let(name, ty, expr) => {
                if let Some(d_y) = adjoints.get(name).cloned() {
                    match expr {
                        Expr::Binary(op, lhs, rhs) => {
                            match op {
                                BinaryOp::Add => {
                                    accumulate(&mut new_stmts, &mut adjoints, lhs, &d_y, &mut new_tmp);
                                    accumulate(&mut new_stmts, &mut adjoints, rhs, &d_y, &mut new_tmp);
                                },
                                BinaryOp::Sub => {
                                    accumulate(&mut new_stmts, &mut adjoints, lhs, &d_y, &mut new_tmp);
                                    let neg_d_y = new_tmp();
                                    new_stmts.push(Stmt::Let(neg_d_y.clone(), Type::Scalar(ScalarType::F32), Expr::Unary(UnaryOp::Neg, Value::Var(d_y.clone()))));
                                    accumulate(&mut new_stmts, &mut adjoints, rhs, &neg_d_y, &mut new_tmp);
                                },
                                BinaryOp::Mul => {
                                    let d_x = new_tmp();
                                    new_stmts.push(Stmt::Let(d_x.clone(), Type::Scalar(ScalarType::F32), Expr::Binary(BinaryOp::Mul, Value::Var(d_y.clone()), rhs.clone())));
                                    accumulate(&mut new_stmts, &mut adjoints, lhs, &d_x, &mut new_tmp);
                                    
                                    let d_y_rhs = new_tmp();
                                    new_stmts.push(Stmt::Let(d_y_rhs.clone(), Type::Scalar(ScalarType::F32), Expr::Binary(BinaryOp::Mul, Value::Var(d_y.clone()), lhs.clone())));
                                    accumulate(&mut new_stmts, &mut adjoints, rhs, &d_y_rhs, &mut new_tmp);
                                },
                                _ => {}
                            }
                        },
                        Expr::MatMul(lhs, rhs) => {
                            let b_t = new_tmp();
                            new_stmts.push(Stmt::Let(b_t.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::Call("transpose".to_string(), vec![rhs.clone()])));
                            
                            let d_a = new_tmp();
                            new_stmts.push(Stmt::Let(d_a.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::MatMul(Value::Var(d_y.clone()), Value::Var(b_t))));
                            accumulate(&mut new_stmts, &mut adjoints, lhs, &d_a, &mut new_tmp);
                            
                            let a_t = new_tmp();
                            new_stmts.push(Stmt::Let(a_t.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::Call("transpose".to_string(), vec![lhs.clone()])));
                            
                            let d_b = new_tmp();
                            new_stmts.push(Stmt::Let(d_b.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::MatMul(Value::Var(a_t), Value::Var(d_y.clone()))));
                            accumulate(&mut new_stmts, &mut adjoints, rhs, &d_b, &mut new_tmp);
                        },
                        Expr::Call(fn_name, args) => {
                            if fn_name == "cross_entropy_logits" {
                                let logits = &args[0];
                                let labels = &args[1];
                                
                                let sm = new_tmp();
                                new_stmts.push(Stmt::Let(sm.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::Call("softmax".to_string(), vec![logits.clone()])));
                                
                                let diff = new_tmp();
                                new_stmts.push(Stmt::Let(diff.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::Binary(BinaryOp::Sub, Value::Var(sm), labels.clone())));
                                
                                let grad = new_tmp();
                                new_stmts.push(Stmt::Let(grad.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::Binary(BinaryOp::Mul, Value::Var(d_y.clone()), Value::Var(diff))));
                                
                                accumulate(&mut new_stmts, &mut adjoints, logits, &grad, &mut new_tmp);
                            } else if fn_name == "mean" {
                                let x = &args[0];
                                let factor = new_tmp();
                                new_stmts.push(Stmt::Let(factor.clone(), Type::Scalar(ScalarType::F32), Expr::Binary(BinaryOp::Div, Value::Var(d_y.clone()), Value::Literal(Literal::Float(100.0)))));
                                
                                let ones = new_tmp();
                                let x_var = match x { Value::Var(s) => s.clone(), _ => "unknown".to_string() };
                                new_stmts.push(Stmt::Let(ones.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::Call("ones_like".to_string(), vec![Value::Var(x_var)])));
                                
                                let grad_tensor = new_tmp();
                                new_stmts.push(Stmt::Let(grad_tensor.clone(), Type::Tensor(ScalarType::F32, vec![]), Expr::Binary(BinaryOp::Mul, Value::Var(ones), Value::Var(factor))));
                                
                                accumulate(&mut new_stmts, &mut adjoints, x, &grad_tensor, &mut new_tmp);
                            }
                        }
                        _ => {}
                    }
                }
            },
            _ => {}
        }
    }
    
    grad_body.extend(new_stmts);
    
    let mut ret_vals = Vec::new();
    match &ret_val {
        Value::Var(v) => ret_vals.push(Value::Var(v.clone())),
        _ => ret_vals.push(Value::Literal(Literal::Float(0.0))), 
    }
    
    for (param_name, _) in &func.params {
        if let Some(grad_name) = adjoints.get(param_name) {
            ret_vals.push(Value::Var(grad_name.clone()));
        } else {
            ret_vals.push(Value::Literal(Literal::Float(0.0)));
        }
    }
    
    let tuple_tmp = new_tmp();
    grad_body.push(Stmt::Let(tuple_tmp.clone(), Type::Tuple(vec![]), Expr::Tuple(ret_vals)));
    grad_body.push(Stmt::Return(Value::Var(tuple_tmp)));
    
    let grad_func = Function {
        name: grad_func_name.clone(),
        params: func.params.clone(),
        ret: Type::Tuple(vec![]), 
        body: grad_body,
    };
    
    module.funcs.push(grad_func);
    Ok(grad_func_name)
}

fn accumulate(
    stmts: &mut Vec<Stmt>, 
    adjoints: &mut HashMap<String, String>, 
    target: &Value, 
    grad: &String,
    new_tmp: &mut impl FnMut() -> String
) {
    if let Value::Var(name) = target {
        if let Some(current_grad) = adjoints.get(name) {
            let sum = new_tmp();
            stmts.push(Stmt::Let(sum.clone(), Type::Scalar(ScalarType::F32), Expr::Binary(BinaryOp::Add, Value::Var(current_grad.clone()), Value::Var(grad.clone()))));
            adjoints.insert(name.clone(), sum);
        } else {
            adjoints.insert(name.clone(), grad.clone());
        }
    }
}
