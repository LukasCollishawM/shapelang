use chumsky::prelude::*;
use crate::ast::*;
use crate::lexer::Token;

pub fn parser() -> impl Parser<Token, Vec<Spanned<Stmt>>, Error = Simple<Token>> {
    let expr = recursive(|expr| {
        let val = select! {
            Token::Int(n) => Expr::Literal(Literal::Int(n)),
            Token::Float(n) => Expr::Literal(Literal::Float(n)),
            Token::True => Expr::Literal(Literal::Bool(true)),
            Token::False => Expr::Literal(Literal::Bool(false)),
            Token::Ident(s) => Expr::Var(s),
            Token::String(s) => Expr::String(s),
        };

        let atom = val
            .or(expr.clone().delimited_by(just(Token::LParen), just(Token::RParen)));

        // Postfix operations: Call and Member access
        // They are left associative: a.b.c()
        
        let postfix = atom.then(
            just(Token::Dot).ignore_then(select! { Token::Ident(s) => s })
                .map(move |field| (false, field, vec![]))
            .or(
                expr.clone().separated_by(just(Token::Comma)).allow_trailing().delimited_by(just(Token::LParen), just(Token::RParen))
                .map(|args| (true, String::new(), args))
            )
            .repeated()
        ).foldl(|lhs, (is_call, field, args)| {
            if is_call {
                Expr::Call(Box::new(lhs), args)
            } else {
                Expr::Member(Box::new(lhs), field)
            }
        });

        let product = postfix.clone()
            .then(just(Token::Star).or(just(Token::Slash)).then(postfix).repeated())
            .foldl(|lhs, (op, rhs)| {
                let op = match op {
                    Token::Star => BinaryOp::Mul,
                    Token::Slash => BinaryOp::Div,
                    _ => unreachable!(),
                };
                Expr::Binary(Box::new(lhs), op, Box::new(rhs))
            });

        let sum = product.clone()
            .then(just(Token::Plus).or(just(Token::Minus)).then(product).repeated())
            .foldl(|lhs, (op, rhs)| {
                let op = match op {
                    Token::Plus => BinaryOp::Add,
                    Token::Minus => BinaryOp::Sub,
                    _ => unreachable!(),
                };
                Expr::Binary(Box::new(lhs), op, Box::new(rhs))
            });

        sum
    });

    let type_parser = recursive(|type_p| {
        let scalar = select! {
            Token::F32 => ScalarType::F32,
            Token::I64 => ScalarType::I64,
            Token::Bool => ScalarType::Bool,
        };

        let shape_dim = select! {
            Token::Int(n) => ShapeDim::Fixed(n as usize),
            Token::Ident(s) => ShapeDim::Symbol(s),
        };

        let tensor = just(Token::Tensor)
            .ignore_then(just(Token::LBracket))
            .ignore_then(scalar.clone())
            .then_ignore(just(Token::Comma))
            .then(shape_dim.separated_by(just(Token::Comma)).delimited_by(just(Token::LBracket), just(Token::RBracket)))
            .then_ignore(just(Token::RBracket))
            .map(|(dt, shape)| Type::Tensor(dt, shape));

        let tuple = type_p.separated_by(just(Token::Comma))
            .delimited_by(just(Token::LParen), just(Token::RParen))
            .map(Type::Tuple);

        tensor.or(scalar.map(Type::Scalar)).or(tuple)
    });

    let stmt = recursive(|stmt| {
        let block = stmt.clone().repeated().delimited_by(just(Token::LBrace), just(Token::RBrace));

        let let_stmt = just(Token::Let)
            .ignore_then(select! { Token::Ident(s) => s })
            .then(just(Token::Colon).ignore_then(type_parser.clone()).or_not())
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then_ignore(just(Token::Semi))
            .map(|((name, ty), e)| Stmt::Let(name, ty, e));

        let return_stmt = just(Token::Return)
            .ignore_then(expr.clone())
            .then_ignore(just(Token::Semi))
            .map(Stmt::Return);
        
        let expr_stmt = expr.clone()
            .then_ignore(just(Token::Semi))
            .map(Stmt::Expr);

        let fn_def = just(Token::Fn)
            .ignore_then(select! { Token::Ident(s) => s })
            .then(
                select! { Token::Ident(s) => s }
                    .then_ignore(just(Token::Colon))
                    .then(type_parser.clone())
                    .separated_by(just(Token::Comma))
                    .delimited_by(just(Token::LParen), just(Token::RParen))
            )
            .then_ignore(just(Token::Arrow))
            .then(type_parser.clone())
            .then(block.clone())
            .map(|(((name, args), ret_ty), body)| Stmt::FnDef(name, args, ret_ty, body));

        let for_loop = just(Token::For)
            .ignore_then(select! { Token::Ident(s) => s })
            .then_ignore(just(Token::In))
            .then_ignore(just(Token::Int(0))) 
            .then_ignore(just(Token::DotDot))
            .then(select! { Token::Int(n) => n as usize })
            .then(block.clone())
            .map(|((var, n), body)| Stmt::For(var, n, Box::new(body)));

        let block_expr = stmt.clone().repeated().delimited_by(just(Token::LBrace), just(Token::RBrace))
            .map(Expr::Block);

        let if_stmt = just(Token::If)
            .ignore_then(expr.clone())
            .then(block_expr.clone().map(|b| if let Expr::Block(stmts) = b { stmts } else { vec![] }))
            .then(just(Token::Else).ignore_then(block_expr.clone().map(|b| if let Expr::Block(stmts) = b { stmts } else { vec![] })).or_not())
            .map(|((cond, then_block), else_block)| {
                 Stmt::Expr(Expr::If(Box::new(cond), Box::new(then_block), Box::new(else_block.unwrap_or_default())))
            });
            
        let import_stmt = just(Token::Import)
            .ignore_then(select! { Token::Ident(s) => s })
            .then_ignore(just(Token::Semi))
            .map(Stmt::Import);

        fn_def
            .or(let_stmt)
            .or(return_stmt)
            .or(for_loop)
            .or(if_stmt)
            .or(import_stmt)
            .or(expr_stmt)
            .map_with_span(|node, span| Spanned { node, span: span.start..span.end })
    });

    stmt.repeated().then_ignore(end())
}
