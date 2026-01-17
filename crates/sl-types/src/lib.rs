use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ScalarType {
    F32,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShapeDim {
    Fixed(usize),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Scalar(ScalarType),
    Tensor(ScalarType, Vec<ShapeDim>),
    Fn(Vec<Type>, Box<Type>),
    Tuple(Vec<Type>),
}

#[derive(Debug, Clone)]
pub struct Constraint {
    pub lhs: ShapeDim,
    pub rhs: ShapeDim,
}

#[derive(Debug)]
pub struct TypeError {
    pub message: String,
    // span: Range<usize>, // To add later
}

pub struct TypeEnv {
    pub vars: HashMap<String, Type>,
    pub shapes: HashMap<String, usize>, // Resolved symbolic shapes
}

impl TypeEnv {
    pub fn new() -> Self {
        Self {
            vars: HashMap::new(),
            shapes: HashMap::new(),
        }
    }

    pub fn insert(&mut self, name: String, ty: Type) {
        self.vars.insert(name, ty);
    }

    pub fn lookup(&self, name: &str) -> Option<&Type> {
        self.vars.get(name)
    }
}

// Unification logic for shapes
pub fn unify_shapes(s1: &[ShapeDim], s2: &[ShapeDim], env: &mut TypeEnv) -> Result<(), TypeError> {
    if s1.len() != s2.len() {
        return Err(TypeError {
            message: format!("Rank mismatch: {} != {}", s1.len(), s2.len()),
        });
    }

    for (d1, d2) in s1.iter().zip(s2.iter()) {
        match (d1, d2) {
            (ShapeDim::Fixed(n1), ShapeDim::Fixed(n2)) => {
                if n1 != n2 {
                    return Err(TypeError {
                        message: format!("Dimension mismatch: {} != {}", n1, n2),
                    });
                }
            }
            (ShapeDim::Symbol(sym), ShapeDim::Fixed(n)) | (ShapeDim::Fixed(n), ShapeDim::Symbol(sym)) => {
                // Check if symbol is already resolved
                if let Some(&val) = env.shapes.get(sym) {
                    if val != *n {
                        return Err(TypeError {
                            message: format!("Symbol {} mismatch: expected {}, found {}", sym, val, n),
                        });
                    }
                } else {
                    // Bind symbol
                    env.shapes.insert(sym.clone(), *n);
                }
            }
            (ShapeDim::Symbol(sym1), ShapeDim::Symbol(sym2)) => {
                if sym1 != sym2 {
                     return Err(TypeError {
                        message: format!("Symbolic mismatch: {} != {}", sym1, sym2),
                    });
                }
            }
        }
    }
    Ok(())
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::Scalar(s) => write!(f, "{:?}", s),
            Type::Tensor(dt, shape) => {
                write!(f, "Tensor[{:?}, [", dt)?;
                for (i, dim) in shape.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    match dim {
                        ShapeDim::Fixed(n) => write!(f, "{}", n)?,
                        ShapeDim::Symbol(s) => write!(f, "{}", s)?,
                    }
                }
                write!(f, "]]")
            }
            Type::Fn(args, ret) => {
                write!(f, "fn(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", arg)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", t)?;
                }
                write!(f, ")")
            }
        }
    }
}
