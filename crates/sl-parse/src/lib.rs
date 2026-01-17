pub mod ast;
pub mod lexer;
pub mod parser;

use lexer::Token;
use logos::Logos;
use chumsky::Parser;

pub fn parse(input: &str) -> Result<Vec<ast::Spanned<ast::Stmt>>, Vec<String>> {
    let lexer = Token::lexer(input);
    let tokens: Vec<(Token, std::ops::Range<usize>)> = lexer.spanned().map(|(tok, span)| (tok, span)).collect();
    
    // Check for lexer errors
    for (tok, _) in &tokens {
        if matches!(tok, Token::Error) {
             return Err(vec!["Lexer error".to_string()]);
        }
    }

    let token_stream = chumsky::Stream::from_iter(
        tokens.len()..tokens.len() + 1,
        tokens.into_iter(),
    );

    parser::parser().parse(token_stream).map_err(|errs| {
        errs.into_iter().map(|e| format!("{:?}", e)).collect()
    })
}
