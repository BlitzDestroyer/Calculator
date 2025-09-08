use thiserror::Error;

use crate::debug_println;

#[derive(Debug, PartialEq, Clone)]
pub enum LexicalToken {
    Integer(u64),
    Float(f64),
    Identifier(String),
    StringLiteral(String),
    Boolean(bool),
    Plus,
    Minus,
    Asterisk,
    SlashForward,
    SlashBackward,
    ParenthesisLeft,
    ParenthesisRight,
    BracketLeft,
    BracketRight,
    BraceLeft,
    BraceRight,
    PercentSign,
    Caret,
    Ampersand,
    ExclamationMark,
    Pipe,
    Tilde,
    EqualSign,
    GreaterThanSign,
    LessThanSign,
    EqualEqualSign,
    ExclamationMarkEqualSign,
    LessThanEqualSign,
    GreaterThanEqualSign,
    GreaterThanGreaterThanSign,
    LessThanLessThanSign,
    Colon,
    ColonColon,
    Semicolon,
    Comma,
    Newline,
    If,
    Else,
    While,
    For,
    Loop,
}

impl LexicalToken {
    pub fn get_static_literal(&self) -> Option<&'static str> {
        match self {
            LexicalToken::Integer(_) | LexicalToken::Float(_) | LexicalToken::Identifier(_) | LexicalToken::StringLiteral(_) => None,
            LexicalToken::Boolean(value) => Some(if *value { "true" } else { "false" }),
            LexicalToken::Plus => Some("+"),
            LexicalToken::Minus => Some("-"),
            LexicalToken::Asterisk => Some("*"),
            LexicalToken::SlashForward => Some("/"),
            LexicalToken::SlashBackward => Some("\\"),
            LexicalToken::ParenthesisLeft => Some("("),
            LexicalToken::ParenthesisRight => Some(")"),
            LexicalToken::BracketLeft => Some("["),
            LexicalToken::BracketRight => Some("]"),
            LexicalToken::BraceLeft => Some("{"),
            LexicalToken::BraceRight => Some("}"),
            LexicalToken::PercentSign => Some("%"),
            LexicalToken::Caret => Some("^"),
            LexicalToken::Ampersand => Some("&"),
            LexicalToken::ExclamationMark => Some("!"),
            LexicalToken::Pipe => Some("|"),
            LexicalToken::Tilde => Some("~"),
            LexicalToken::EqualSign => Some("="),
            LexicalToken::GreaterThanSign => Some(">"),
            LexicalToken::LessThanSign => Some("<"),
            LexicalToken::Newline => Some("\n"),
            LexicalToken::EqualEqualSign => Some("=="),
            LexicalToken::ExclamationMarkEqualSign => Some("!="),
            LexicalToken::LessThanEqualSign => Some("<="),
            LexicalToken::GreaterThanEqualSign => Some(">="),
            LexicalToken::GreaterThanGreaterThanSign => Some(">>"),
            LexicalToken::LessThanLessThanSign => Some("<<"),
            LexicalToken::Colon => Some(":"),
            LexicalToken::ColonColon => Some("::"),
            LexicalToken::Semicolon => Some(";"),
            LexicalToken::Comma => Some(","),
            LexicalToken::If => Some("if"),
            LexicalToken::Else => Some("else"),
            LexicalToken::While => Some("while"),
            LexicalToken::For => Some("for"),
            LexicalToken::Loop => Some("loop"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Context {
    token_literal: String, // How the token appears in the source code
    line: usize,
    column: usize,
}

impl Context {
    pub fn new(token_literal: String, line: usize, column: usize) -> Self {
        Self { token_literal, line, column }
    }

    pub fn get_token_literal(&self) -> &str {
        &self.token_literal
    }

    pub fn get_line(&self) -> usize {
        self.line
    }

    pub fn get_column(&self) -> usize {
        self.column
    }

    pub fn get_position(&self) -> (usize, usize) {
        (self.line, self.column)
    }
}

#[derive(Debug, Clone)]
pub struct LexicalTokenContext {
    token: LexicalToken,
    context: Context,
}

impl LexicalTokenContext {
    pub fn new(token: LexicalToken, token_literal: String, line: usize, column: usize) -> Self {
        Self { token, context: Context::new(token_literal, line, column) }
    }

    pub fn new_static(token: LexicalToken, line: usize, column: usize) -> Self {
        let token_literal = token.get_static_literal();
        assert_ne!(token_literal, None);
        let token_literal = token_literal.unwrap().to_string();
        Self { token, context: Context::new(token_literal, line, column) }
    }

    pub fn get_token(&self) -> &LexicalToken {
        &self.token
    }

    pub fn get_context(&self) -> &Context {
        &self.context
    }
}

impl PartialEq for LexicalTokenContext {
    fn eq(&self, other: &Self) -> bool {
        self.token == other.token
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum LexicalTokenType {
    None,
    Int,
    Hex,
    Octal,
    Binary,
    Float,
    Identifier,
    StringLiteral,
}

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
}

#[derive(Debug)]
pub struct Lexer {
    tokens: Vec<LexicalTokenContext>,
    index: usize,
}

impl Lexer {
    pub fn new(input: &str) -> Result<Self, LexicalTokenizeError> {
        let tokens = tokenize(input)?;
        Ok(Self { tokens, index: 0 })
    }

    pub fn next(&mut self) -> Option<&LexicalTokenContext> {
        let curr_index = self.index;
        self.index += 1;
        self.tokens.get(curr_index)
    }

    pub fn peek(&self) -> Option<&LexicalTokenContext> {
        self.tokens.get(self.index)
    }
}

pub fn tokenize(input: &str) -> Result<Vec<LexicalTokenContext>, LexicalTokenizeError> {
    let chars = input.chars();
    let mut tokens = Vec::new();
    let mut token_type = LexicalTokenType::None;
    let mut buffer = String::new();
    let mut escaped = false;
    let mut char_pos = 0; // Set to 0 as it gets incremented at the start of the loop
    let mut line_num = 1;
    for c in chars {
        char_pos += 1;
        debug_println!("Char: '{}', TokenType: {:?}, Buffer: '{}'", c, token_type, buffer);
        match c {
            c if c >= '0' && c <= '1' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                if token_type == LexicalTokenType::None {
                    token_type = LexicalTokenType::Int;
                }

                buffer.push(c);
            }
            c if c >= '2' && c <= '7' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                match token_type {
                    LexicalTokenType::None => token_type = LexicalTokenType::Int,
                    LexicalTokenType::Binary => return Err(LexicalTokenizeError::InvalidBinaryLiteral),
                    _ => {},
                }

                buffer.push(c);
            }
            c if c >= '8' && c <= '9' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                match token_type {
                    LexicalTokenType::None => token_type = LexicalTokenType::Int,
                    LexicalTokenType::Binary => return Err(LexicalTokenizeError::InvalidBinaryLiteral),
                    LexicalTokenType::Octal => return Err(LexicalTokenizeError::InvalidOctalLiteral),
                    _ => {}
                }

                buffer.push(c);
            }
            c if (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                token_type = match token_type {
                    LexicalTokenType::None => LexicalTokenType::Identifier,
                    LexicalTokenType::Int => { // Handles both 0b<binary> and hex chars
                        if c == 'b' {
                            if buffer.as_str() == "0" && token_type == LexicalTokenType::Int {
                                LexicalTokenType::Binary
                            }
                            else if token_type == LexicalTokenType::StringLiteral{
                                LexicalTokenType::StringLiteral
                            }
                            else{
                                LexicalTokenType::Identifier
                            }
                        }
                        else{
                            LexicalTokenType::Identifier // Switch to identifier if alpha char found after int
                        }
                    },
                    LexicalTokenType::Octal => return Err(LexicalTokenizeError::InvalidOctalLiteral),
                    LexicalTokenType::Binary => return Err(LexicalTokenizeError::InvalidBinaryLiteral),
                    _ => token_type,
                };

                buffer.push(c); // Used to separate alpha hex chars from default case
            }
            'x' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                token_type = match token_type {
                    LexicalTokenType::None => LexicalTokenType::Identifier,
                    LexicalTokenType::Int if buffer.as_str() == "0" => LexicalTokenType::Hex,
                    LexicalTokenType::StringLiteral => LexicalTokenType::StringLiteral,
                    LexicalTokenType::Int => return Err(LexicalTokenizeError::UnexpectedCharacter(c, (line_num, char_pos), line!())),
                    _ => LexicalTokenType::Identifier
                };

                buffer.push(c);
            }
            'o' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                token_type = match token_type {
                    LexicalTokenType::None => LexicalTokenType::Identifier,
                    LexicalTokenType::Int if buffer.as_str() == "0" => LexicalTokenType::Octal,
                    LexicalTokenType::StringLiteral => LexicalTokenType::StringLiteral,
                    LexicalTokenType::Int => return Err(LexicalTokenizeError::UnexpectedCharacter(c, (line_num, char_pos), line!())),
                    _ => LexicalTokenType::Identifier
                };

                buffer.push(c);
            }
            '.' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                match token_type {
                    LexicalTokenType::None => return Err(LexicalTokenizeError::UnexpectedCharacter(c, (line_num, char_pos), line!())),
                    LexicalTokenType::Int => token_type = LexicalTokenType::Float,
                    LexicalTokenType::Hex | LexicalTokenType::Octal | LexicalTokenType::Binary => token_type = LexicalTokenType::Identifier,
                    _ => (),
                }

                buffer.push(c);
            }
            '+' => {
                lex_simple_token(LexicalToken::Plus, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '-' => {
                lex_simple_token(LexicalToken::Minus, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '*' => {
                lex_simple_token(LexicalToken::Asterisk, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '/' => {
                lex_simple_token(LexicalToken::SlashForward, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '(' => {
                lex_simple_token(LexicalToken::ParenthesisLeft, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            ')' => {
                lex_simple_token(LexicalToken::ParenthesisRight, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '[' => {
                lex_simple_token(LexicalToken::BracketLeft, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            ']' => {
                lex_simple_token(LexicalToken::BracketRight, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '{' => {
                lex_simple_token(LexicalToken::BraceLeft, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '}' => {
                lex_simple_token(LexicalToken::BraceRight, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '%' => {
                lex_simple_token(LexicalToken::PercentSign, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '^' => {
                lex_simple_token(LexicalToken::Caret, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '&' => {
                lex_simple_token(LexicalToken::Ampersand, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '|' => {
                lex_simple_token(LexicalToken::Pipe, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '!' => {
                lex_simple_token(LexicalToken::ExclamationMark, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '~' => {
                lex_simple_token(LexicalToken::Tilde, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            ';' => {
                lex_simple_token(LexicalToken::Semicolon, &mut buffer, &mut token_type, &mut tokens, escaped, c, line_num, char_pos, line!())?;
            }
            '>' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                parse_buffer(&mut buffer, token_type, &mut tokens, line_num, char_pos)?;
                let prev_token = tokens.last();
                let (token, must_pop) = if let Some(prev_token) = prev_token {
                    match prev_token.token {
                        LexicalToken::GreaterThanSign => (LexicalToken::GreaterThanGreaterThanSign, true),
                        _ => (LexicalToken::GreaterThanSign, false),
                    }
                }
                else{
                    (LexicalToken::GreaterThanSign, false)
                };

                // If need to remove prev token to form new double token
                if must_pop {
                    tokens.pop();
                }

                let token = LexicalTokenContext::new_static(token, line_num, char_pos - buffer.len());
                tokens.push(token);
                token_type = LexicalTokenType::None;
            }
            '<' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                parse_buffer(&mut buffer, token_type, &mut tokens, line_num, char_pos)?;
                let prev_token = tokens.last();
                let (token, must_pop) = if let Some(prev_token) = prev_token {
                    match prev_token.token {
                        LexicalToken::LessThanSign => (LexicalToken::LessThanLessThanSign, true),
                        _ => (LexicalToken::LessThanSign, false),
                    }
                }
                else{
                    (LexicalToken::LessThanSign, false)
                };

                // If need to remove prev token to form new double token
                if must_pop {
                    tokens.pop();
                }

                let token = LexicalTokenContext::new_static(token, line_num, char_pos - buffer.len());
                tokens.push(token);
                token_type = LexicalTokenType::None;
            }
            '=' => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }

                parse_buffer(&mut buffer, token_type, &mut tokens, line_num, char_pos)?;
                let prev_token = tokens.last();
                let (token, must_pop) = if let Some(prev_token) = prev_token {
                    match prev_token.token {
                        LexicalToken::EqualSign => (LexicalToken::EqualEqualSign, true),
                        LexicalToken::ExclamationMark => (LexicalToken::ExclamationMarkEqualSign, true),
                        LexicalToken::LessThanSign => (LexicalToken::LessThanEqualSign, true),
                        LexicalToken::GreaterThanSign => (LexicalToken::GreaterThanEqualSign, true),
                        _ => (LexicalToken::EqualSign, false),
                    }
                }
                else{
                    (LexicalToken::EqualSign, false)
                };

                // If need to remove prev token to form new double token
                if must_pop {
                    tokens.pop();
                }

                let token = LexicalTokenContext::new_static(token, line_num, char_pos - buffer.len());
                tokens.push(token);
                token_type = LexicalTokenType::None;
            }
            '"' => {
                match token_type {
                    LexicalTokenType::None if !escaped => { // Start of string literal
                        token_type = LexicalTokenType::StringLiteral;
                    }
                    LexicalTokenType::StringLiteral if !escaped => { // End of string literal
                        
                        let token = LexicalTokenContext::new(LexicalToken::StringLiteral(buffer.clone()), buffer.clone(), line_num, char_pos - buffer.len());
                        tokens.push(token);
                        buffer.clear();
                        token_type = LexicalTokenType::None;
                    }
                    LexicalTokenType::StringLiteral if escaped => { // Inside string literal
                        buffer.push('"');
                        escaped = false;
                    }
                    _ => return Err(LexicalTokenizeError::UnexpectedCharacter(c, (line_num, char_pos), line!()))
                }
            },
            '\\' => {
                if escaped {
                    buffer.push('\\');
                    escaped = false;
                }
                else {
                    escaped = true;
                }
            }
            'n' => {
                if escaped {
                    match token_type {
                        LexicalTokenType::StringLiteral => buffer.push('\n'),
                        _ => {
                            parse_buffer(&mut buffer, token_type, &mut tokens, line_num, char_pos)?;
                            token_type = LexicalTokenType::None;
                            let token_literal = LexicalToken::Newline.get_static_literal().unwrap().to_string();
                            let token = LexicalTokenContext::new(LexicalToken::Newline, token_literal, line_num, char_pos - buffer.len());
                            tokens.push(token);
                            char_pos = 0; // Gets incremented at the start of the loop
                            line_num += 1;
                        },
                    }
                    escaped = false;
                }
                else {
                    buffer.push('n');
                }
            }
            c if c.is_whitespace() => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }
                
                if token_type != LexicalTokenType::StringLiteral {
                    parse_buffer(&mut buffer, token_type, &mut tokens, line_num, char_pos)?;
                    token_type = LexicalTokenType::None;
                }
                else {
                    buffer.push(c);
                }
            }
            c => {
                if escaped {
                    return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), line!()));
                }
                
                if token_type != LexicalTokenType::StringLiteral {
                    token_type = LexicalTokenType::Identifier;
                }

                buffer.push(c);
            }
        }
    }

    // Flush buffer
    parse_buffer(&mut buffer, token_type, &mut tokens, line_num, char_pos)?;

    Ok(tokens)
}

fn lex_simple_token(simple_token: LexicalToken, buffer: &mut String, token_type: &mut LexicalTokenType, tokens: &mut Vec<LexicalTokenContext>, escaped: bool, c: char, line_num: usize, char_pos: usize, source_line: u32) -> Result<(), LexicalTokenizeError> {
    if escaped {
        return Err(LexicalTokenizeError::UnknownEscapeSequence(c, (line_num, char_pos), source_line));
    }
    
    match *token_type {
        LexicalTokenType::None => {
            let token_literal = simple_token.get_static_literal().unwrap().to_string();
            let token = LexicalTokenContext::new(simple_token, token_literal, line_num, char_pos);
            tokens.push(token);
            *token_type = LexicalTokenType::None;
        },
        LexicalTokenType::Int | LexicalTokenType::Hex | LexicalTokenType::Octal | LexicalTokenType::Binary | LexicalTokenType::Float | LexicalTokenType::Identifier => {
            parse_buffer(buffer, *token_type, tokens, line_num, char_pos)?;
            let token_literal = simple_token.get_static_literal().unwrap().to_string();
            let token = LexicalTokenContext::new(simple_token, token_literal, line_num, char_pos - buffer.len());
            tokens.push(token);
            *token_type = LexicalTokenType::None;
        },
        LexicalTokenType::StringLiteral => buffer.push(c),
    }

    Ok(())
}

fn parse_buffer(buffer: &mut String, token_type: LexicalTokenType, tokens: &mut Vec<LexicalTokenContext>, line_num: usize, char_pos: usize) -> Result<(), LexicalTokenizeError>{
    if !buffer.is_empty() {
        match token_type {
            LexicalTokenType::Int => {
                let value = buffer.parse::<u64>()?;
                let token = LexicalToken::Integer(value);
                let token_literal = buffer.clone();
                let token = LexicalTokenContext::new(token, token_literal, line_num, char_pos - buffer.len());
                tokens.push(token);
            },
            LexicalTokenType::Float => {
                let value = buffer.parse::<f64>()?;
                let token = LexicalToken::Float(value);
                let token_literal = buffer.clone();
                let token = LexicalTokenContext::new(token, token_literal, line_num, char_pos - buffer.len());
                tokens.push(token);
            },
            LexicalTokenType::Identifier => {
                let token = match buffer.as_str() {
                    "true" => LexicalToken::Boolean(true),
                    "false" => LexicalToken::Boolean(false),
                    "if" => LexicalToken::If,
                    "else" => LexicalToken::Else,
                    "while" => LexicalToken::While,
                    "for" => LexicalToken::For,
                    "loop" => LexicalToken::Loop,
                    _ => LexicalToken::Identifier(buffer.clone()),
                };
                let token_literal = buffer.clone();
                let token = LexicalTokenContext::new(token, token_literal, line_num, char_pos - buffer.len());
                tokens.push(token);
            },
            LexicalTokenType::Hex => {
                let value = u64::from_str_radix(&buffer[2..], 16)?;
                let token = LexicalToken::Integer(value);
                let token_literal = buffer.clone();
                let token = LexicalTokenContext::new(token, token_literal, line_num, char_pos - buffer.len());
                tokens.push(token);
            },
            LexicalTokenType::Octal => {
                let value = u64::from_str_radix(&buffer[2..], 8)?;
                let token = LexicalToken::Integer(value);
                let token_literal = buffer.clone();
                let token = LexicalTokenContext::new(token, token_literal, line_num, char_pos - buffer.len());
                tokens.push(token);
            },
            LexicalTokenType::Binary => {
                let value = u64::from_str_radix(&buffer[2..], 2)?;
                let token = LexicalToken::Integer(value);
                let token_literal = buffer.clone();
                let token = LexicalTokenContext::new(token, token_literal, line_num, char_pos - buffer.len());
                tokens.push(token);
            },
            LexicalTokenType::StringLiteral => {
                return Err(LexicalTokenizeError::UnterminatedStringLiteral);
            },
            LexicalTokenType::None => return Err(LexicalTokenizeError::TokenTypeCannotBeNoneWhenBufferIsNotEmpty),
        }

        buffer.clear();
    }

    Ok(())
}

#[cfg(test)]
mod lexical_test {
    use super::*;

    #[test]
    fn test_tokenization_int() {
        let input = "5";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 5);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_float() {
        let input = "3.14";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Float(value) = &tokens[0].token {
            assert_eq!(*value, 3.14);
        }
        else {
            panic!("Expected float token");
        }
    }

    #[test]
    fn test_tokenization_identifier() {
        let input = "hello";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Identifier(value) = &tokens[0].token {
            assert_eq!(value, "hello");
        }
        else {
            panic!("Expected string token");
        }
    }

    #[test]
    fn test_tokenization_string_literal_1() {
        let input = "\"hello\"";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::StringLiteral(value) = &tokens[0].token {
            assert_eq!(value, "hello");
        }
        else {
            panic!("Expected string token");
        }
    }

    #[test]
    fn test_tokenization_string_literal_2() {
        let input = "\"The quick brown fox jumps over the lazy dog\"";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::StringLiteral(value) = &tokens[0].token {
            assert_eq!(value, "The quick brown fox jumps over the lazy dog");
        }
        else {
            panic!("Expected string token");
        }
    }
    
    #[test]
    fn test_tokenization_hex_1() {
        let input = "0x12345678";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 0x12345678);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_hex_2() {
        let input = "0x9ABCDEF";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 0x9ABCDEF);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_string_literal_3() {
        let input = "\"0x1234f\"";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::StringLiteral(value) = &tokens[0].token {
            assert_eq!(value, "0x1234f");
        }
        else {
            panic!("Expected string token");
        }
    }

    #[test]
    fn test_tokenization_binary_1() {
        let input = "0b0";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 0b0);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_binary_2() {
        let input = "0b101";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 0b101);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_binary_3() {
        let input = "0b101010";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 0b101010);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_octal_1() {
        let input = "0o10";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 0o10);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_octal_2() {
        let input = "0o77";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Integer(value) = &tokens[0].token {
            assert_eq!(*value, 0o77);
        }
        else {
            panic!("Expected integer token");
        }
    }

    #[test]
    fn test_tokenization_bool_1() {
        let input = "true";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Boolean(value) = &tokens[0].token {
            assert_eq!(*value, true);
        }
        else {
            panic!("Expected boolean token");
        }
    }

    #[test]
    fn test_tokenization_bool_2() {
        let input = "false";
        let tokens = tokenize(input).unwrap();
        assert_eq!(tokens.len(), 1);
        if let LexicalToken::Boolean(value) = &tokens[0].token {
            assert_eq!(*value, false);
        }
        else {
            panic!("Expected boolean token");
        }
    }

    #[test]
    fn test_tokenization_if_else() {
        let input = "if { } else { };";
        let tokens = tokenize(input).unwrap();
        let expected = vec![
            LexicalToken::If,
            LexicalToken::BraceLeft,
            LexicalToken::BraceRight,
            LexicalToken::Else,
            LexicalToken::BraceLeft,
            LexicalToken::BraceRight,
            LexicalToken::Semicolon,
        ];
        assert_eq!(tokens.len(), expected.len());
        for (i, token) in tokens.iter().enumerate() {
            assert_eq!(&token.token, &expected[i]);
        }
    }

    #[test]
    fn test_tokenization_compound_1() {
        let input = "1+23-345/678*9.1011";
        let tokens = tokenize(input).unwrap();
        let expected = vec![
            LexicalToken::Integer(1),
            LexicalToken::Plus,
            LexicalToken::Integer(23),
            LexicalToken::Minus,
            LexicalToken::Integer(345),
            LexicalToken::SlashForward,
            LexicalToken::Integer(678),
            LexicalToken::Asterisk,
            LexicalToken::Float(9.1011),
        ];
        assert_eq!(tokens.len(), expected.len());
        for (i, token) in tokens.iter().enumerate() {
            assert_eq!(&token.token, &expected[i]);
        }
    }
}