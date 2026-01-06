use thiserror::Error;

use crate::lexing::{LexicalTokenContext, engine::{LexicalTokenizeError, TokenStream}};

#[derive(Debug, Error)]
pub enum AstError {
    #[error("Lexical tokenize error: {0}")]
    LexicalTokenizeError(#[from] LexicalTokenizeError),
    #[error("Unexpected token")]
    UnexpectedToken(LexicalTokenContext, u32), // Added u32 to provide caller context
    #[error("Expected operator")]
    ExpectedOperator(LexicalTokenContext),
    #[error("Unmatched parentheses")]
    UnmatchedParentheses,
    #[error("Unmatched brackets")]
    UnmatchedBrackets,
    #[error("Unmatched braces")]
    UnmatchedBraces,
    #[error("Unexpected end of file")]
    UnexpectedEndOfFile,
}

pub trait Grammar {
    type Token;
    type Ast;
    type Error;

    fn atom(token: &Self::Token) -> Option<Self::Ast>;
    fn unexpected_end_of_file() -> Self::Error;
    fn unexpected_token(token: &Self::Token, source_line: u32) -> Self::Error;

    fn prefix_binding_power(op: &Self::Ast, token: &Self::Token) -> Result<u8, Self::Error>;
    fn infix_binding_power(op: &Self::Ast, token: &Self::Token) -> Result<Option<(u8, u8)>, Self::Error>;
    fn postfix_binding_power(op: &Self::Ast, token: &Self::Token) -> Result<Option<u8>, Self::Error>;

    fn led(op: Self::Ast, args: Vec<Self::Ast>) -> Result<Self::Ast, Self::Error>;

    fn led_postfix<TS: TokenStream<Self::Token>>(
        ast: Self::Ast,
        op: Self::Token,
        lexer: &mut TS
    ) -> Result<Self::Ast, Self::Error>;

    fn nud<TS: TokenStream<Self::Token>>(
        head: Self::Ast,
        head_token: Self::Token,
        lexer: &mut TS
    ) -> Result<Self::Ast, Self::Error>;
}

pub fn parse<G: Grammar, TS: TokenStream<G::Token>>(
    lexer: &mut TS,
    min_bp: u8,
) -> Result<G::Ast, G::Error> {
    let mut lhs = {
        let token = lexer.next().ok_or(G::unexpected_end_of_file())?;
        let lhs = G::atom(&token)
            .ok_or_else(|| G::unexpected_token(&token, line!()))?;

        G::nud(lhs, token, lexer)?
    };

    loop {
        let Some(peek) = lexer.peek() else { break };
        // Prob need a function here to check if peek is an operator or delimiter

        let op = G::atom(&peek)
            .ok_or_else(|| G::unexpected_token(&peek, line!()))?;
        if let Some(l_bp) = G::postfix_binding_power(&op, peek)? {
            if l_bp < min_bp { break }
            // Safety: we just checked that peek is Some
            let op = lexer.next().unwrap(); 
            // prob need to change this to handle postfix operator as the earlier call has different semantics/functioanlity
            lhs = G::led_postfix(lhs, op, lexer)?;

            continue;
        }

        if let Some((l_bp, r_bp)) = G::infix_binding_power(&op, peek)? {
            if l_bp < min_bp { break }
            lexer.next();
            let rhs = parse::<G, TS>(lexer, r_bp)?;
            lhs = G::led(op, vec![lhs, rhs])?;

            continue;
        }

        break;
    }

    Ok(lhs)
}
