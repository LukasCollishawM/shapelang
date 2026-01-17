use sl_types::{Type, ScalarType, ShapeDim, TypeEnv, TypeError};
use sl_parse::ast::{self, BinaryOp};

#[derive(Debug, Clone)]
pub struct Module {
    pub funcs: Vec<Function>,
}

#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<(String, Type)>,
    pub ret: Type,
    pub body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
pub enum Stmt {
    Let(String, Type, Expr),
    Return(Value),
    Expr(Expr),
    If(Value, Vec<Stmt>, Vec<Stmt>),
    For(String, usize, Vec<Stmt>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Binary(BinaryOp, Value, Value),
    Unary(UnaryOp, Value),
    Call(String, Vec<Value>),
    Literal(Literal),
    MatMul(Value, Value),
    Builtin(String, Vec<Value>), 
    Grad(String),
    Tuple(Vec<Value>), 
    TupleGet(Value, usize),
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnaryOp {
    Neg, Not, Exp, Log, Relu, Tanh,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Value {
    Var(String),
    Literal(Literal),
    String(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
}

impl std::hash::Hash for Literal {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Literal::Int(i) => i.hash(state),
            Literal::Float(f) => f.to_bits().hash(state),
            Literal::Bool(b) => b.hash(state),
        }
    }
}

impl Eq for Literal {}

pub struct LoweringCtx {
    env: TypeEnv,
    tmp_counter: usize,
}

impl LoweringCtx {
    pub fn new() -> Self {
        Self { env: TypeEnv::new(), tmp_counter: 0 }
    }

    fn new_tmp(&mut self) -> String {
        self.tmp_counter += 1;
        format!("_t{}", self.tmp_counter)
    }

    pub fn lower_module(&mut self, stmts: Vec<ast::Spanned<ast::Stmt>>) -> Result<Module, String> {
        let mut funcs = Vec::new();
        let mut main_body = Vec::new();

        for stmt in stmts {
            match stmt.node {
                ast::Stmt::FnDef(name, args, ret_ty, body) => {
                    let ir_args: Vec<(String, Type)> = args.into_iter()
                        .map(|(n, t)| (n, self.convert_type(t)))
                        .collect();
                    
                    for (n, t) in &ir_args {
                        self.env.insert(n.clone(), t.clone());
                    }

                    let ir_ret = self.convert_type(ret_ty);
                    let ir_body = self.lower_block(body)?;
                    funcs.push(Function {
                        name,
                        params: ir_args,
                        ret: ir_ret,
                        body: ir_body,
                    });
                },
                ast::Stmt::Import(_) => { /* Ignore import */ },
                _ => {
                    let ir_stmts = self.lower_stmt(stmt.node)?;
                    main_body.extend(ir_stmts);
                }
            }
        }
        
        if !main_body.is_empty() {
             funcs.push(Function {
                name: "main".to_string(),
                params: vec![],
                ret: Type::Scalar(ScalarType::I64),
                body: main_body,
            });
        }

        Ok(Module { funcs })
    }

    fn convert_type(&self, t: ast::Type) -> Type {
        match t {
            ast::Type::Scalar(s) => match s {
                ast::ScalarType::F32 => Type::Scalar(ScalarType::F32),
                ast::ScalarType::I64 => Type::Scalar(ScalarType::I64),
                ast::ScalarType::Bool => Type::Scalar(ScalarType::Bool),
            },
            ast::Type::Tensor(s, dims) => {
                let st = match s {
                    ast::ScalarType::F32 => ScalarType::F32,
                    ast::ScalarType::I64 => ScalarType::I64,
                    ast::ScalarType::Bool => ScalarType::Bool,
                };
                let d = dims.into_iter().map(|d| match d {
                    ast::ShapeDim::Fixed(n) => ShapeDim::Fixed(n),
                    ast::ShapeDim::Symbol(s) => ShapeDim::Symbol(s),
                }).collect();
                Type::Tensor(st, d)
            },
            ast::Type::Tuple(ts) => {
                Type::Tuple(ts.into_iter().map(|t| self.convert_type(t)).collect())
            }
        }
    }

    fn lower_block(&mut self, stmts: Vec<ast::Spanned<ast::Stmt>>) -> Result<Vec<Stmt>, String> {
        let mut res = Vec::new();
        for stmt in stmts {
            res.extend(self.lower_stmt(stmt.node)?);
        }
        Ok(res)
    }

    fn lower_stmt(&mut self, stmt: ast::Stmt) -> Result<Vec<Stmt>, String> {
        match stmt {
            ast::Stmt::Let(name, ty_opt, expr) => {
                let mut stmts = Vec::new();
                let (val, expr_ty) = self.lower_expr_to_value(expr, &mut stmts)?;
                let ty = if let Some(t) = ty_opt { self.convert_type(t) } else { expr_ty };
                self.env.insert(name.clone(), ty.clone());
                match val {
                    Value::Var(v) => {
                         // Emit identity op via addition with zero to alias the variable
                         stmts.push(Stmt::Let(name, ty, Expr::Binary(BinaryOp::Add, Value::Var(v), Value::Literal(Literal::Int(0))))); 
                         Ok(stmts) 
                    },
                    Value::Literal(l) => {
                        stmts.push(Stmt::Let(name, ty, Expr::Literal(l)));
                        Ok(stmts)
                    },
                    Value::String(_) => {
                         Err("String literals only allowed in function arguments".into())
                    }
                }
            },
            ast::Stmt::Return(e) => {
                let mut stmts = Vec::new();
                let (val, _) = self.lower_expr_to_value(e, &mut stmts)?;
                stmts.push(Stmt::Return(val));
                Ok(stmts)
            },
            ast::Stmt::Expr(e) => {
                let mut stmts = Vec::new();
                let _ = self.lower_expr_to_value(e, &mut stmts)?;
                Ok(stmts)
            },
            ast::Stmt::Import(_) => Ok(vec![]),
            ast::Stmt::FnDef(..) => Err("Nested functions not supported".to_string()),
            ast::Stmt::For(var, n, body) => {
                 self.env.insert(var.clone(), Type::Scalar(ScalarType::I64));
                 let ir_body = self.lower_block(*body)?;
                 Ok(vec![Stmt::For(var, n, ir_body)])
            }
        }
    }

    fn lower_expr_to_value(&mut self, expr: ast::Expr, stmts: &mut Vec<Stmt>) -> Result<(Value, Type), String> {
        match expr {
            ast::Expr::Var(n) => {
                let ty = self.env.lookup(&n).ok_or_else(|| format!("Unknown var {}", n))?.clone();
                Ok((Value::Var(n), ty))
            },
            ast::Expr::Literal(l) => {
                 let ty = match l {
                     ast::Literal::Int(_) => Type::Scalar(ScalarType::I64),
                     ast::Literal::Float(_) => Type::Scalar(ScalarType::F32),
                     ast::Literal::Bool(_) => Type::Scalar(ScalarType::Bool),
                 };
                 let lit = match l {
                     ast::Literal::Int(v) => Literal::Int(v),
                     ast::Literal::Float(v) => Literal::Float(v),
                     ast::Literal::Bool(v) => Literal::Bool(v),
                 };
                 Ok((Value::Literal(lit), ty))
            },
            ast::Expr::String(s) => {
                Ok((Value::String(s), Type::Scalar(ScalarType::I64)))
            },
            ast::Expr::Binary(lhs, op, rhs) => {
                let (l_val, l_ty) = self.lower_expr_to_value(*lhs, stmts)?;
                let (r_val, _r_ty) = self.lower_expr_to_value(*rhs, stmts)?;
                let res_ty = l_ty.clone(); 
                let tmp = self.new_tmp();
                stmts.push(Stmt::Let(tmp.clone(), res_ty.clone(), Expr::Binary(op, l_val, r_val)));
                Ok((Value::Var(tmp), res_ty))
            },
            ast::Expr::Call(callee, args) => {
                let (func_name, extra_arg_expr) = match *callee {
                    ast::Expr::Var(n) => (n, None),
                    ast::Expr::Member(obj, field) => {
                         if let ast::Expr::Var(ref v) = *obj {
                             if v == "dataset" {
                                 (format!("dataset.{}", field), None)
                             } else {
                                 (field, Some(*obj)) 
                             }
                         } else {
                             (field, Some(*obj))
                         }
                    },
                    _ => return Err("Indirect call not supported".into()),
                };

                let mut ir_args = Vec::new();
                
                if let Some(arg) = extra_arg_expr {
                    let (v_val, _) = self.lower_expr_to_value(arg, stmts)?;
                    ir_args.push(v_val);
                }

                for arg in args {
                     let (v, _) = self.lower_expr_to_value(arg, stmts)?;
                     ir_args.push(v);
                }
                
                if func_name == "grad" {
                    if ir_args.len() != 1 { return Err("grad takes 1 arg".into()); }
                    if let Value::Var(fn_name_val) = &ir_args[0] {
                        let tmp = self.new_tmp();
                        let ty = Type::Scalar(ScalarType::I64); 
                        stmts.push(Stmt::Let(tmp.clone(), ty.clone(), Expr::Grad(fn_name_val.clone())));
                        Ok((Value::Var(tmp), ty))
                    } else {
                        Err("grad arg must be function".into())
                    }
                } else if func_name == "matmul" {
                     let ret_ty = Type::Tensor(ScalarType::F32, vec![]); // Simplified infer
                     let tmp = self.new_tmp();
                     stmts.push(Stmt::Let(tmp.clone(), ret_ty.clone(), Expr::MatMul(ir_args[0].clone(), ir_args[1].clone())));
                     Ok((Value::Var(tmp), ret_ty))
                } else {
                     let tmp = self.new_tmp();
                     let ty = Type::Tensor(ScalarType::F32, vec![]); 
                     stmts.push(Stmt::Let(tmp.clone(), ty.clone(), Expr::Call(func_name, ir_args)));
                     Ok((Value::Var(tmp), ty))
                }
            },
            ast::Expr::Member(obj, field) => {
                 // Property access? `ds.batches`?
                 // Or just fail if not a call.
                 Err("Property access not supported".into())
            },
            ast::Expr::If(cond, then_block, else_block) => {
                 let (c_val, _) = self.lower_expr_to_value(*cond, stmts)?;
                 let then_ir = self.lower_block(*then_block)?;
                 let else_ir = self.lower_block(*else_block)?;
                 stmts.push(Stmt::If(c_val, then_ir, else_ir));
                 Ok((Value::Literal(Literal::Int(0)), Type::Scalar(ScalarType::I64))) 
            },
            ast::Expr::Block(stmts_block) => {
                 let ir_stmts = self.lower_block(stmts_block)?;
                 stmts.extend(ir_stmts);
                 Ok((Value::Literal(Literal::Int(0)), Type::Scalar(ScalarType::I64))) 
            },
             _ => Err("Not implemented".to_string())
        }
    }

    fn infer_matmul(&self, t1: &Type, t2: &Type) -> Result<Type, String> {
        if let (Type::Tensor(dt1, s1), Type::Tensor(dt2, s2)) = (t1, t2) {
             if dt1 != dt2 { return Err("Dtype mismatch".into()); }
             Ok(Type::Tensor(dt1.clone(), vec![s1[0].clone(), s2[1].clone()]))
        } else {
            Ok(Type::Tensor(ScalarType::F32, vec![]))
        }
    }
}
