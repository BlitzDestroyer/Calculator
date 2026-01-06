pub mod engine;

use crate::{debug_println, lexing::engine::{LexerAction, LexerSpec, LexicalTokenizeError, TokenStream}};

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
    AmpersandAmpersand,
    ExclamationMark,
    Pipe,
    PipePipe,
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
    Break,
    Continue,
    Let,
    Const,
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
            LexicalToken::AmpersandAmpersand => Some("&&"),
            LexicalToken::ExclamationMark => Some("!"),
            LexicalToken::Pipe => Some("|"),
            LexicalToken::PipePipe => Some("||"),
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
            LexicalToken::Break => Some("break"),
            LexicalToken::Continue => Some("continue"),
            LexicalToken::Let => Some("let"),
            LexicalToken::Const => Some("const"),
        }
    }

    pub fn from_static_literal(s: &str) -> Option<Self> {
        match s {
            "true" => Some(LexicalToken::Boolean(true)),
            "false" => Some(LexicalToken::Boolean(false)),
            "+" => Some(LexicalToken::Plus),
            "-" => Some(LexicalToken::Minus),
            "*" => Some(LexicalToken::Asterisk),
            "/" => Some(LexicalToken::SlashForward),
            "\\" => Some(LexicalToken::SlashBackward),
            "(" => Some(LexicalToken::ParenthesisLeft),
            ")" => Some(LexicalToken::ParenthesisRight),
            "[" => Some(LexicalToken::BracketLeft),
            "]" => Some(LexicalToken::BracketRight),
            "{" => Some(LexicalToken::BraceLeft),
            "}" => Some(LexicalToken::BraceRight),
            "%" => Some(LexicalToken::PercentSign),
            "^" => Some(LexicalToken::Caret),
            "&" => Some(LexicalToken::Ampersand),
            "&&" => Some(LexicalToken::AmpersandAmpersand),
            "!" => Some(LexicalToken::ExclamationMark),
            "|" => Some(LexicalToken::Pipe),
            "||" => Some(LexicalToken::PipePipe),
            "~" => Some(LexicalToken::Tilde),
            "=" => Some(LexicalToken::EqualSign),
            ">" => Some(LexicalToken::GreaterThanSign),
            "<" => Some(LexicalToken::LessThanSign),
            "\n" => Some(LexicalToken::Newline),
            "==" => Some(LexicalToken::EqualEqualSign),
            "!=" => Some(LexicalToken::ExclamationMarkEqualSign),
            "<=" => Some(LexicalToken::LessThanEqualSign),
            ">=" => Some(LexicalToken::GreaterThanEqualSign),
            ">>" => Some(LexicalToken::GreaterThanGreaterThanSign),
            "<<" => Some(LexicalToken::LessThanLessThanSign),
            ":" => Some(LexicalToken::Colon),
            "::" => Some(LexicalToken::ColonColon),
            ";" => Some(LexicalToken::Semicolon),
            "," => Some(LexicalToken::Comma),
            "if" => Some(LexicalToken::If),
            "else" => Some(LexicalToken::Else),
            "while" => Some(LexicalToken::While),
            "for" => Some(LexicalToken::For),
            "loop" => Some(LexicalToken::Loop),
            "break" => Some(LexicalToken::Break),
            "continue" => Some(LexicalToken::Continue),
            "let" => Some(LexicalToken::Let),
            "const" => Some(LexicalToken::Const),
            _ => None,
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
pub enum LexicalTokenType {
    None,
    Int,
    Hex,
    Octal,
    Binary,
    Float,
    Identifier,
    StringLiteral,
    PendingAmpersand,
    PendingPipe,
    PendingGreaterThanSign,
    PendingLessThanSign,
    PendingEqualSign,
    PendingExclamationMark,
    PendingColon,
}

#[derive(Debug)]
pub struct Lexer {
    tokens: Vec<Option<LexicalTokenContext>>,
    index: usize,
}

impl LexerSpec for Lexer {
    type Token = LexicalTokenContext;
    type State = LexicalTokenType;

    fn initial_state() -> Self::State {
        LexicalTokenType::None
    }

    fn step(
        state: Self::State,
        c: char,
        buffer: &str,
        escaped: &mut bool,
        pos: (usize, usize)
    ) -> engine::LexerAction<Self::State> {
        match c {
            c if c >= '0' && c <= '1' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                let state = if state == LexicalTokenType::None {
                    LexicalTokenType::Int
                }
                else {
                    state
                };

                LexerAction::Continue(state)
            }
            c if c >= '2' && c <= '7' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                let state = match state {
                    LexicalTokenType::None => LexicalTokenType::Int,
                    LexicalTokenType::Binary => return LexerAction::Error(LexicalTokenizeError::InvalidBinaryLiteral),
                    _ => state,
                };

                LexerAction::Continue(state)
            }
            c if c >= '8' && c <= '9' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                let state = match state {
                    LexicalTokenType::None => LexicalTokenType::Int,
                    LexicalTokenType::Binary => return LexerAction::Error(LexicalTokenizeError::InvalidBinaryLiteral),
                    LexicalTokenType::Octal => return LexerAction::Error(LexicalTokenizeError::InvalidOctalLiteral),
                    _ => state,
                };

                LexerAction::Continue(state)
            }
            c if (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                let state = match state {
                    LexicalTokenType::None => LexicalTokenType::Identifier,
                    LexicalTokenType::Int => { // Handles both 0b<binary> and hex chars
                        if c == 'b' {
                            if buffer == "0" && state == LexicalTokenType::Int {
                                LexicalTokenType::Binary
                            }
                            else if state == LexicalTokenType::StringLiteral{
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
                    LexicalTokenType::Octal => return LexerAction::Error(LexicalTokenizeError::InvalidOctalLiteral),
                    LexicalTokenType::Binary => return LexerAction::Error(LexicalTokenizeError::InvalidBinaryLiteral),
                    _ => state,
                };

                LexerAction::Continue(state) // Used to separate alpha hex chars from default case
            }
            'x' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                let state = match state {
                    LexicalTokenType::None => LexicalTokenType::Identifier,
                    LexicalTokenType::Int if buffer == "0" => LexicalTokenType::Hex,
                    LexicalTokenType::StringLiteral => LexicalTokenType::StringLiteral,
                    LexicalTokenType::Int => return LexerAction::Error(LexicalTokenizeError::UnexpectedCharacter(c, pos, line!())),
                    _ => LexicalTokenType::Identifier
                };

                LexerAction::Continue(state)
            }
            'o' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                let state = match state {
                    LexicalTokenType::None => LexicalTokenType::Identifier,
                    LexicalTokenType::Int if buffer == "0" => LexicalTokenType::Octal,
                    LexicalTokenType::StringLiteral => LexicalTokenType::StringLiteral,
                    LexicalTokenType::Int => return LexerAction::Error(LexicalTokenizeError::UnexpectedCharacter(c, pos, line!())),
                    _ => LexicalTokenType::Identifier
                };

                LexerAction::Continue(state)
            }
            '.' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                let state = match state {
                    LexicalTokenType::None => return LexerAction::Error(LexicalTokenizeError::UnexpectedCharacter(c, pos, line!())),
                    LexicalTokenType::Int => LexicalTokenType::Float,
                    LexicalTokenType::Hex | LexicalTokenType::Octal | LexicalTokenType::Binary => LexicalTokenType::Identifier,
                    _ => state,
                };

                LexerAction::Continue(state)
            }
            '+' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '-' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '*' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '/' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '(' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            ')' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '[' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            ']' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '{' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '}' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '%' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '^' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '&' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                match state {
                    LexicalTokenType::PendingAmpersand => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::AmpersandAmpersand.get_static_literal().unwrap())
                    }
                    _ => {
                        LexerAction::Transition(LexicalTokenType::PendingAmpersand)
                    }
                }
            }
            '|' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                match state {
                    LexicalTokenType::PendingPipe => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::PipePipe.get_static_literal().unwrap())
                    }
                    _ => {
                        LexerAction::Transition(LexicalTokenType::PendingPipe)
                    }
                }
            }
            '!' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                LexerAction::Transition(LexicalTokenType::PendingExclamationMark)
            }
            '~' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            ';' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '>' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                match state {
                    LexicalTokenType::PendingGreaterThanSign => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::GreaterThanGreaterThanSign.get_static_literal().unwrap())
                    }
                    _ => {
                        LexerAction::Transition(LexicalTokenType::PendingGreaterThanSign)
                    }
                }
            }
            '<' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                match state {
                    LexicalTokenType::PendingLessThanSign => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::LessThanLessThanSign.get_static_literal().unwrap())
                    }
                    _ => {
                        LexerAction::Transition(LexicalTokenType::PendingLessThanSign)
                    }
                }
            }
            '=' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                match state {
                    LexicalTokenType::PendingEqualSign => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::EqualEqualSign.get_static_literal().unwrap())
                    }
                    LexicalTokenType::PendingExclamationMark => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::ExclamationMarkEqualSign.get_static_literal().unwrap())
                    }
                    LexicalTokenType::PendingGreaterThanSign => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::GreaterThanEqualSign.get_static_literal().unwrap())
                    }
                    LexicalTokenType::PendingLessThanSign => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::LessThanEqualSign.get_static_literal().unwrap())
                    }
                    _ => {
                        LexerAction::Transition(LexicalTokenType::PendingEqualSign)
                    }
                }
            }
            ':' => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }

                match state {
                    LexicalTokenType::PendingColon => {
                        LexerAction::EmitStrTokenAndReset(LexicalToken::ColonColon.get_static_literal().unwrap())
                    }
                    _ => {
                        LexerAction::Transition(LexicalTokenType::PendingColon)
                    }
                }
            }
            ',' => {
                lex_simple_token(state, *escaped, c, pos, line!())
            }
            '"' => {
                match state {
                    LexicalTokenType::None if !*escaped => { // Start of string literal
                        LexerAction::Transition(LexicalTokenType::StringLiteral)
                    }
                    LexicalTokenType::StringLiteral if !*escaped => { // End of string literal
                        LexerAction::EmitBufferThen(LexicalTokenType::None)
                    }
                    LexicalTokenType::StringLiteral if *escaped => { // Inside string literal
                        *escaped = false;
                        LexerAction::Continue(state)
                    }
                    _ => return LexerAction::Error(LexicalTokenizeError::UnexpectedCharacter(c, pos, line!()))
                }
            },
            '\\' => {
                if *escaped {
                    *escaped = false;
                    LexerAction::Continue(state)
                }
                else {
                    *escaped = true;
                    LexerAction::Transition(state)
                }
            }
            'n' => {
                if *escaped {
                    *escaped = false;
                    match state {
                        LexicalTokenType::StringLiteral => LexerAction::ContinueWithChar(state, '\n'),
                        _ => {
                            // TODO: Maybe use a different error to better communicate that escape sequences are only valid in string literals
                            LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()))
                        },
                    }
                }
                else {
                    LexerAction::Continue(state)
                }
            }
            c if c.is_whitespace() => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }
                
                if state != LexicalTokenType::StringLiteral {
                    LexerAction::EmitBufferThen(LexicalTokenType::None)
                }
                else {
                    LexerAction::Continue(state)
                }
            }
            c => {
                if *escaped {
                    return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, line!()));
                }
                
                if state != LexicalTokenType::StringLiteral {
                    LexerAction::Continue(LexicalTokenType::Identifier)
                }
                else{
                    LexerAction::Continue(state)
                }
            }
        }
    }

    fn emit_token(
        state: Self::State,
        buffer: &str,
        pos: (usize, usize),
    ) -> Result<Option<Self::Token>, engine::LexicalTokenizeError> {
        match state {
            LexicalTokenType::None => {
                let token = LexicalToken::from_static_literal(buffer);
                match token {
                    Some(tok) => {
                        debug_println!("Emitting static token {:?} for buffer '{}' at pos {:?}", tok, buffer, pos);
                        let token = LexicalTokenContext::new(tok, buffer.to_string(), pos.0, pos.1 - buffer.len());
                        Ok(Some(token))
                    },
                    //None => Err(LexicalTokenizeError::ExpectedTokenButFoundNone(pos)),
                    None => Ok(None),
                }
            },
            LexicalTokenType::Int => {
                debug_println!("Emitting token for buffer '{}' with type Int at pos {:?}", buffer, pos);
                let value = buffer.parse::<u64>()?;
                let token = LexicalTokenContext::new(LexicalToken::Integer(value), buffer.to_string(), pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::Hex => {
                debug_println!("Emitting token for buffer '{}' with type Hex at pos {:?}", buffer, pos);
                let value = u64::from_str_radix(buffer.trim_start_matches("0x"), 16)?;
                let token = LexicalTokenContext::new(LexicalToken::Integer(value), buffer.to_string(), pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::Octal => {
                debug_println!("Emitting token for buffer '{}' with type Octal at pos {:?}", buffer, pos);
                let value = u64::from_str_radix(buffer.trim_start_matches("0o"), 8)?;
                let token = LexicalTokenContext::new(LexicalToken::Integer(value), buffer.to_string(), pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::Binary => {
                debug_println!("Emitting token for buffer '{}' with type Binary at pos {:?}", buffer, pos);
                let value = u64::from_str_radix(buffer.trim_start_matches("0b"), 2)?;
                let token = LexicalTokenContext::new(LexicalToken::Integer(value), buffer.to_string(), pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::Float => {
                debug_println!("Emitting token for buffer '{}' with type Float at pos {:?}", buffer, pos);
                let value = buffer.parse::<f64>()?;
                let token = LexicalTokenContext::new(LexicalToken::Float(value), buffer.to_string(), pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::Identifier => {
                debug_println!("Emitting token for buffer '{}' with type Identifier at pos {:?}", buffer, pos);
                let token = match buffer {
                    "true" => LexicalToken::Boolean(true),
                    "false" => LexicalToken::Boolean(false),
                    "if" => LexicalToken::If,
                    "else" => LexicalToken::Else,
                    "while" => LexicalToken::While,
                    "for" => LexicalToken::For,
                    "loop" => LexicalToken::Loop,
                    "break" => LexicalToken::Break,
                    "continue" => LexicalToken::Continue,
                    "let" => LexicalToken::Let,
                    "const" => LexicalToken::Const,
                    _ => LexicalToken::Identifier(buffer.to_string()),
                };
                let token = LexicalTokenContext::new(token, buffer.to_string(), pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::StringLiteral => {
                debug_println!("Emitting token for buffer '{}' with type StringLiteral at pos {:?}", buffer, pos);
                let token = LexicalTokenContext::new(LexicalToken::StringLiteral(buffer.to_string()), buffer.to_string(), pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::PendingAmpersand => {
                debug_println!("Emitting token for buffer '{}' with type PendingAmpersand at pos {:?}", buffer, pos);
                let token = LexicalToken::Ampersand;
                let token = LexicalTokenContext::new_static(token, pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::PendingPipe => {
                debug_println!("Emitting token for buffer '{}' with type PendingPipe at pos {:?}", buffer, pos);
                let token = LexicalToken::Pipe;
                let token = LexicalTokenContext::new_static(token, pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::PendingGreaterThanSign => {
                debug_println!("Emitting token for buffer '{}' with type PendingGreaterThanSign at pos {:?}", buffer, pos);
                let token = LexicalToken::GreaterThanSign;
                let token = LexicalTokenContext::new_static(token, pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::PendingLessThanSign => {
                debug_println!("Emitting token for buffer '{}' with type PendingLessThanSign at pos {:?}", buffer, pos);
                let token = LexicalToken::LessThanSign;
                let token = LexicalTokenContext::new_static(token, pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::PendingEqualSign => {
                debug_println!("Emitting token for buffer '{}' with type PendingEqualSign at pos {:?}", buffer, pos);
                let token = LexicalToken::EqualSign;
                let token = LexicalTokenContext::new_static(token, pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::PendingExclamationMark => {
                debug_println!("Emitting token for buffer '{}' with type PendingExclamationMark at pos {:?}", buffer, pos);
                let token = LexicalToken::ExclamationMark;
                let token = LexicalTokenContext::new_static(token, pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
            LexicalTokenType::PendingColon => {
                debug_println!("Emitting token for buffer '{}' with type PendingColon at pos {:?}", buffer, pos);
                let token = LexicalToken::Colon;
                let token = LexicalTokenContext::new_static(token, pos.0, pos.1 - buffer.len());
                Ok(Some(token))
            },
        }
    }

    fn finalize(
        state: Self::State,
        buffer: &str,
        pos: (usize, usize),
    ) -> Result<Option<Self::Token>, LexicalTokenizeError> {
        match state {
            LexicalTokenType::StringLiteral => {
                Err(LexicalTokenizeError::UnterminatedStringLiteral)
            }
            _ => Self::emit_token(state, buffer, pos)
        }
    }
}

impl TokenStream<LexicalTokenContext> for Lexer {
    fn next(&mut self) -> Option<LexicalTokenContext> {
        if self.index < self.tokens.len() {
            let token = self.tokens[self.index].take();
            self.index += 1;
            token
        } 
        else {
            None
        }
    }

    fn peek(&self) -> Option<&LexicalTokenContext> {
        self.tokens.get(self.index).map(|t| t.as_ref()).flatten()
    }
}

impl Lexer {
    pub fn new(input: &str) -> Result<Self, LexicalTokenizeError> {
        let tokens = engine::lex::<Self>(input)?;
        let tokens = tokens.into_iter().map(|t| Some(t)).collect();
        Ok(Self { tokens, index: 0 })
    }

    fn next(&mut self) -> Option<&LexicalTokenContext> {
        let curr_index = self.index;
        self.index += 1;
        self.tokens.get(curr_index).unwrap().as_ref()
    }
}

fn lex_simple_token(state: LexicalTokenType, escaped: bool, c: char, pos: (usize, usize), source_line: u32) -> LexerAction<LexicalTokenType> {
    if escaped {
        return LexerAction::Error(LexicalTokenizeError::UnknownEscapeSequence(c, pos, source_line));
    }

    match state {
        LexicalTokenType::None => {
            LexerAction::EmitBufferThenCharToken(LexicalTokenType::None, c)
        }
        LexicalTokenType::Int | LexicalTokenType::Hex | LexicalTokenType::Octal | LexicalTokenType::Binary | LexicalTokenType::Float | LexicalTokenType::Identifier => {
            // First emit buffer as token
            // Then emit simple_token as token
            // Finally return to None state
            LexerAction::EmitBufferThenCharToken(LexicalTokenType::None, c)
        }
        LexicalTokenType::StringLiteral => LexerAction::Continue(state),
        LexicalTokenType::PendingAmpersand
        | LexicalTokenType::PendingPipe
        | LexicalTokenType::PendingGreaterThanSign
        | LexicalTokenType::PendingLessThanSign
        | LexicalTokenType::PendingEqualSign
        | LexicalTokenType::PendingExclamationMark
        | LexicalTokenType::PendingColon => unreachable!(
            "lex_simple_token called with pending operator state: {:?}",
            state
        ),
    }
}

#[cfg(test)]
mod lexical_test {
    use super::*;

    #[test]
    fn test_tokenization_token_dump() {
        let input = "1 2.345 hello \"string literal\" true + - * / \\\\ ( ) [ ] { } % ^ & && ! | || ~ = > < == != <= >= >> << : :: ; , \n if else while for loop break continue let const";
        let expected_tokens = vec![
            LexicalToken::Integer(1),
            LexicalToken::Float(2.345),
            LexicalToken::Identifier("hello".to_string()),
            LexicalToken::StringLiteral("string literal".to_string()),
            LexicalToken::Boolean(true),
            LexicalToken::Plus,
            LexicalToken::Minus,
            LexicalToken::Asterisk,
            LexicalToken::SlashForward,
            LexicalToken::SlashBackward,
            LexicalToken::ParenthesisLeft,
            LexicalToken::ParenthesisRight,
            LexicalToken::BracketLeft,
            LexicalToken::BracketRight,
            LexicalToken::BraceLeft,
            LexicalToken::BraceRight,
            LexicalToken::PercentSign,
            LexicalToken::Caret,
            LexicalToken::Ampersand,
            LexicalToken::AmpersandAmpersand,
            LexicalToken::ExclamationMark,
            LexicalToken::Pipe,
            LexicalToken::PipePipe,
            LexicalToken::Tilde,
            LexicalToken::EqualSign,
            LexicalToken::GreaterThanSign,
            LexicalToken::LessThanSign,
            LexicalToken::EqualEqualSign,
            LexicalToken::ExclamationMarkEqualSign,
            LexicalToken::LessThanEqualSign,
            LexicalToken::GreaterThanEqualSign,
            LexicalToken::GreaterThanGreaterThanSign,
            LexicalToken::LessThanLessThanSign,
            LexicalToken::Colon,
            LexicalToken::ColonColon,
            LexicalToken::Semicolon,
            LexicalToken::Comma,
            //LexicalToken::Newline,
            LexicalToken::If,
            LexicalToken::Else,
            LexicalToken::While,
            LexicalToken::For,
            LexicalToken::Loop,
            LexicalToken::Break,
            LexicalToken::Continue,
            LexicalToken::Let,
            LexicalToken::Const,
        ];

        let tokens = engine::lex::<Lexer>(input).unwrap();
        assert_eq!(tokens.len(), expected_tokens.len(), "Got {:#?}, expected {:#?}", tokens, expected_tokens);
        for (i, expected_token) in expected_tokens.iter().enumerate() {
            assert_eq!(&tokens[i].token, expected_token);
        }
    }

    #[test]
    fn test_tokenization_int() {
        let input = "5";
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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
        let tokens = engine::lex::<Lexer>(input).unwrap();
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

    #[test]
    fn test_conditional() {
        let input = "if x == 0 { 1 } else if x == 1 { 2 } else { 3 }";
        let tokens = engine::lex::<Lexer>(input).unwrap();
        let expected = vec![
            LexicalToken::If,
            LexicalToken::Identifier("x".to_string()),
            LexicalToken::EqualEqualSign,
            LexicalToken::Integer(0),
            LexicalToken::BraceLeft,
            LexicalToken::Integer(1),
            LexicalToken::BraceRight,
            LexicalToken::Else,
            LexicalToken::If,
            LexicalToken::Identifier("x".to_string()),
            LexicalToken::EqualEqualSign,
            LexicalToken::Integer(1),
            LexicalToken::BraceLeft,
            LexicalToken::Integer(2),
            LexicalToken::BraceRight,
            LexicalToken::Else,
            LexicalToken::BraceLeft,
            LexicalToken::Integer(3),
            LexicalToken::BraceRight,
        ];

        assert_eq!(tokens.len(), expected.len(), "Token count mismatch: expected {:#?}, got {:#?}", expected, tokens);
        for (i, token) in tokens.iter().enumerate() {
            assert_eq!(&token.token, &expected[i], "Token mismatch at position {}: expected {:?}, got {:?}", i, expected[i], token.token);
        }
    }
}