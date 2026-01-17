use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    Tensor(ScalarType, Vec<ShapeDim>),
    Scalar(ScalarType),
    Tuple(Vec<Type>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarType {
    F32,
    I64,
    Bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ShapeDim {
    Fixed(usize),
    Symbol(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub node: T,
    pub span: std::ops::Range<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    Let(String, Option<Type>, Expr),
    Expr(Expr),
    FnDef(String, Vec<(String, Type)>, Type, Vec<Spanned<Stmt>>),
    Return(Expr),
    For(String, usize, Box<Block>),
    Import(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Var(String),
    Binary(Box<Expr>, BinaryOp, Box<Expr>),
    Call(Box<Expr>, Vec<Expr>),
    Member(Box<Expr>, String),
    If(Box<Expr>, Box<Block>, Box<Block>),
    Block(Block),
    String(String),
}

pub type Block = Vec<Spanned<Stmt>>;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BinaryOp {
    Add, Sub, Mul, Div,
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
            },
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
