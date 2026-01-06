use thiserror::Error;

use crate::debug_println;

#[derive(Debug, Error)]
pub enum LexicalTokenizeError {
    #[error("ParseIntError: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("ParseFloatError: {0}")]
    ParseFloatError(#[from] std::num::ParseFloatError),
    #[error("Unexpected character '{0}' at ln {line}, col {column}", line = .1.0, column = .1.1)]
    UnexpectedCharacter(char, (usize, usize), u32), // last usize is source line (this file)
    #[error("Unknown escape sequence '\\{0}' at ln {line}, col {column}", line = .1.0, column = .1.1)]
    UnknownEscapeSequence(char, (usize, usize), u32), // last usize is source line (this file)
    #[error("Unterminated string literal")]
    UnterminatedStringLiteral,
    #[error("Invalid binary literal")]
    InvalidBinaryLiteral,
    #[error("Invalid octal literal")]
    InvalidOctalLiteral,
    #[error("Invalid hexadecimal literal")]
    InvalidHexadecimalLiteral,
    #[error("Token type cannot be None when buffer is not empty")]
    TokenTypeCannotBeNoneWhenBufferIsNotEmpty,
    #[error("Expected token, but found none at ln {line}, col {column}", line = .0.0, column = .0.1)]
    ExpectedTokenButFoundNone((usize, usize)),
    #[error("Other error: {0}")]
    Other(String),
}

pub enum LexerAction<S> {
    Transition(S),                              // consume char, don't add to buffer
    Continue(S),                                // consume char, stay in buffer
    ContinueWithChar(S, char),                  // consume char, stay in buffer, but add different char
    EmitBufferThen(S),                          // flush buffer, then transition
    EmitBufferThenCharToken(S, char),           // flush + emit single-char token
    EmitBufferThenStrToken(S, &'static str),    // flush + emit static-str token
    EmitStrTokenAndReset(&'static str),         // emit static-str token, clear buffer, and reset state
    Error(LexicalTokenizeError),
}

pub trait LexerSpec {
    type Token;
    type State: Copy + Eq + std::fmt::Debug;

    fn initial_state() -> Self::State;

    fn step(
        state: Self::State,
        c: char,
        buffer: &str,
        escaped: &mut bool,
        pos: (usize, usize),
    ) -> LexerAction<Self::State>;

    fn emit_token(
        state: Self::State,
        buffer: &str,
        pos: (usize, usize),
    ) -> Result<Option<Self::Token>, LexicalTokenizeError>;

    fn finalize(
        state: Self::State,
        buffer: &str,
        pos: (usize, usize),
    ) -> Result<Option<Self::Token>, LexicalTokenizeError>;
}

pub fn lex<S: LexerSpec>(
    input: &str,
) -> Result<Vec<S::Token>, LexicalTokenizeError> {
    let mut state = S::initial_state();
    let mut buffer = String::new();
    let mut tokens = Vec::new();

    let mut line = 1;
    let mut col = 0;
    let mut escaped = false;

    for c in input.chars() {
        col += 1;

        debug_println!("Lexing char '{}' (ln {}, col {}), current state: {:?}, buffer: '{}'", c, line, col, state, buffer);
        match S::step(state, c, &buffer, &mut escaped, (line, col)) {
            LexerAction::Transition(next) => {
                state = next;
            }

            LexerAction::Continue(next) => {
                buffer.push(c);
                state = next;
            }

            LexerAction::ContinueWithChar(next, ch) => {
                buffer.push(ch);
                state = next;
            }

            LexerAction::EmitBufferThen(next) => {
                if let Some(tok) = S::emit_token(state, &buffer, (line, col))? {
                    tokens.push(tok);
                }

                buffer.clear();
                state = next;
            }

            LexerAction::EmitBufferThenCharToken(next, single) => {
                if let Some(tok) = S::emit_token(state, &buffer, (line, col))? {
                    tokens.push(tok);
                }

                buffer.clear();
                tokens.push(S::emit_token(
                    next,
                    &single.to_string(),
                    (line, col),
                )?.ok_or_else(|| LexicalTokenizeError::ExpectedTokenButFoundNone((line, col)))?);
                state = S::initial_state();
            }

            LexerAction::EmitBufferThenStrToken(next, static_str) => {
                if let Some(tok) = S::emit_token(state, &buffer, (line, col))? {
                    tokens.push(tok);
                }

                buffer.clear();
                tokens.push(S::emit_token(
                    next,
                    static_str,
                    (line, col),
                )?.ok_or_else(|| LexicalTokenizeError::ExpectedTokenButFoundNone((line, col)))?);
                state = S::initial_state();
            }

            LexerAction::EmitStrTokenAndReset(static_str) => {
                tokens.push(S::emit_token(
                    S::initial_state(),
                    static_str,
                    (line, col),
                )?.ok_or_else(|| LexicalTokenizeError::ExpectedTokenButFoundNone((line, col)))?);
                buffer.clear();
                state = S::initial_state();
            }

            LexerAction::Error(e) => return Err(e),
        }

        if c == '\n' {
            line += 1;
            col = 0;
        }
    }

    if let Some(tok) = S::emit_token(state, &buffer, (line, col))? {
        tokens.push(tok);
    }

    Ok(tokens)
}