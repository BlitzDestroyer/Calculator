use thiserror::Error;

use crate::lexing::{Lexer, LexicalToken, LexicalTokenContext, LexicalTokenizeError};

#[derive(Debug)]
pub struct Program {
    pub body: Vec<AstNode>,
}

impl Program {
    pub fn new(body: Vec<AstNode>) -> Self {
        Program { body }
    }
}

impl std::fmt::Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let body_str: Vec<String> = self.body.iter().map(|node| format!("{}", node)).collect();
        write!(f, "{}", body_str.join("\n"))
    }
}

#[derive(Debug)]
pub enum AstNode {
    Atom(Atom),
    Assignment { target: Box<AstNode>, value: Box<AstNode>, constant: bool },
    Expression { head: Box<AstNode>, args: Vec<Box<AstNode>> },
    Condition { test: Box<AstNode>, consequent: Box<AstNode>, alternate: Option<Box<AstNode>> },
}

impl AstNode {
    fn atom(lexical_token: &LexicalToken) -> Option<AstNode> {
        if let Some(atom) = Atom::value(lexical_token) {
            Some(AstNode::Atom(atom))
        }
        else if let Some(atom) = Atom::operator(lexical_token) {
            Some(AstNode::Atom(atom))
        }
        else if let Some(atom) = Atom::identifier(lexical_token) {
            Some(AstNode::Atom(atom))
        }
        else if let Some(atom) = Atom::delimiter(lexical_token) {
            Some(AstNode::Atom(atom))
        }
        else if let Some(atom) = Atom::keyword(lexical_token) {
            Some(AstNode::Atom(atom))
        }
        else {
            None
        }
    }

    fn is_atom(&self) -> bool {
        matches!(self, AstNode::Atom(_))
    }

    fn is_assignment(&self) -> bool {
        matches!(self, AstNode::Assignment { .. })
    }

    fn is_expression(&self) -> bool {
        matches!(self, AstNode::Expression { .. })
    }

    fn is_condition(&self) -> bool {
        matches!(self, AstNode::Condition { .. })
    }

    fn get_atom(&self) -> Option<&Atom> {
        if let AstNode::Atom(atom) = self {
            Some(atom)
        } 
        else {
            None
        }
    }

    fn get_condition(&self) -> Option<(&AstNode, &AstNode, &Option<Box<AstNode>>)> {
        if let AstNode::Condition { test, consequent, alternate } = self {
            Some((test, consequent, alternate))
        } 
        else {
            None
        }
    }
}

impl std::fmt::Display for AstNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AstNode::Atom(atom) => write!(f, "{}", atom),
            AstNode::Assignment { target, value, constant } => write!(f, "({} {} = {})", if *constant { "const" } else { "let" }, target, value),
            AstNode::Expression { head, args } => {
                let args_str: Vec<String> = args.iter().map(|arg| format!("{}", arg)).collect();
                write!(f, "({} {})", head, args_str.join(" "))
            }
            AstNode::Condition { test, consequent, alternate } => {
                if let Some(alt) = alternate {
                    write!(f, "(if {} {{ {} }} else {{ {} }})", test, consequent, alt)
                } 
                else {
                    write!(f, "(if {} {{ {} }})", test, consequent)
                }
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomKind {
    Value,
    Operator,
    Identifier,
    Keyword,
    Delimiter,
}

#[derive(Debug)]
pub enum Atom {
    Value(Value),
    Operator(Operator),
    Identifier(String),
    Keyword(Keyword),
    Delimiter(Delimiter),
}

impl Atom {
    fn value(lexical_token: &LexicalToken) -> Option<Atom> {
        let val = match lexical_token {
            LexicalToken::Integer(value) => Some(Value::Integer(*value)),
            LexicalToken::Float(value) => Some(Value::Float(*value)),
            LexicalToken::StringLiteral(value) => Some(Value::String(value.clone())),
            LexicalToken::Boolean(value) => Some(Value::Boolean(*value)),
            _ => None,
        };

        val.map(|v| Atom::Value(v))
    }

    fn operator(lexical_token: &LexicalToken) -> Option<Atom> {
        let op = match lexical_token {
            LexicalToken::Plus => Some(Operator::Plus),
            LexicalToken::Minus => Some(Operator::Minus),
            LexicalToken::Asterisk => Some(Operator::Multiply),
            LexicalToken::SlashForward => Some(Operator::Divide),
            LexicalToken::PercentSign => Some(Operator::Modulo),
            LexicalToken::LessThanSign => Some(Operator::LessThan),
            LexicalToken::LessThanEqualSign => Some(Operator::LessThanOrEqual),
            LexicalToken::GreaterThanSign => Some(Operator::GreaterThan),
            LexicalToken::GreaterThanEqualSign => Some(Operator::GreaterThanOrEqual),
            LexicalToken::EqualEqualSign => Some(Operator::Equal),
            LexicalToken::ExclamationMarkEqualSign => Some(Operator::NotEqual),
            LexicalToken::Ampersand => Some(Operator::BitwiseAnd),
            LexicalToken::AmpersandAmpersand => Some(Operator::LogicalAnd),
            LexicalToken::Pipe => Some(Operator::BitwiseOr),
            LexicalToken::PipePipe => Some(Operator::LogicalOr),
            LexicalToken::ExclamationMark => Some(Operator::LogicalNot),
            LexicalToken::Caret => Some(Operator::BitwiseXor),
            LexicalToken::Tilde => Some(Operator::BitwiseNot),
            LexicalToken::LessThanLessThanSign => Some(Operator::LeftShift),
            LexicalToken::GreaterThanGreaterThanSign => Some(Operator::RightShift),
            _ => None,
        };

        op.map(|o| Atom::Operator(o))
    }

    fn identifier(lexical_token: &LexicalToken) -> Option<Atom> {
        if let LexicalToken::Identifier(name) = lexical_token {
            Some(Atom::Identifier(name.clone()))
        } 
        else {
            None
        }
    }

    fn delimiter(lexical_token: &LexicalToken) -> Option<Atom> {
        let delim = match lexical_token {
            LexicalToken::ParenthesisLeft => Some(Delimiter::ParenthesisLeft),
            LexicalToken::ParenthesisRight => Some(Delimiter::ParenthesisRight),
            LexicalToken::BracketLeft => Some(Delimiter::BracketLeft),
            LexicalToken::BracketRight => Some(Delimiter::BracketRight),
            LexicalToken::BraceLeft => Some(Delimiter::BraceLeft),
            LexicalToken::BraceRight => Some(Delimiter::BraceRight),
            _ => None,
        };

        delim.map(|d| Atom::Delimiter(d))
    }

    fn keyword(lexical_token: &LexicalToken) -> Option<Atom> {
        let keyword = match lexical_token {
            LexicalToken::If => Some(Keyword::If),
            LexicalToken::Else => Some(Keyword::Else),
            LexicalToken::While => Some(Keyword::While),
            LexicalToken::For => Some(Keyword::For),
            LexicalToken::Loop => Some(Keyword::Loop),
            LexicalToken::Break => Some(Keyword::Break),
            LexicalToken::Continue => Some(Keyword::Continue),
            LexicalToken::Let => Some(Keyword::Let),
            LexicalToken::Const => Some(Keyword::Const),
            _ => None,
        };

        keyword.map(|k| Atom::Keyword(k))
    }

    fn kind(&self) -> AtomKind {
        match self {
            Atom::Value(_) => AtomKind::Value,
            Atom::Operator(_) => AtomKind::Operator,
            Atom::Identifier(_) => AtomKind::Identifier,
            Atom::Keyword(_) => AtomKind::Keyword,
            Atom::Delimiter(_) => AtomKind::Delimiter,
        }
    }

    fn is_value(&self) -> bool {
        matches!(self, Atom::Value(_))
    }

    fn is_operator(&self) -> bool {
        matches!(self, Atom::Operator(_))
    }

    fn is_identifier(&self) -> bool {
        matches!(self, Atom::Identifier(_))
    }

    fn is_delimiter(&self) -> bool {
        matches!(self, Atom::Delimiter(_))
    }

    fn is_keyword(&self) -> bool {
        matches!(self, Atom::Keyword(_))
    }

    fn get_value(&self) -> Option<&Value> {
        if let Atom::Value(value) = self {
            Some(value)
        }
        else {
            None
        }
    }

    fn get_operator(&self) -> Option<&Operator> {
        if let Atom::Operator(operator) = self {
            Some(operator)
        }
        else {
            None
        }
    }

    fn get_identifier(&self) -> Option<&String> {
        if let Atom::Identifier(name) = self {
            Some(name)
        }
        else {
            None
        }
    }

    fn get_delimiter(&self) -> Option<&Delimiter> {
        if let Atom::Delimiter(delimiter) = self {
            Some(delimiter)
        }
        else {
            None
        }
    }

    fn get_keyword(&self) -> Option<&Keyword> {
        if let Atom::Keyword(keyword) = self {
            Some(keyword)
        }
        else {
            None
        }
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atom::Value(value) => write!(f, "{}", value),
            Atom::Operator(operator) => write!(f, "{}", operator),
            Atom::Identifier(name) => write!(f, "{}", name),
            Atom::Keyword(keyword) => write!(f, "{}", keyword),
            Atom::Delimiter(delimiter) => write!(f, "{}", delimiter),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Unit,
    Integer(u64),
    Float(f64),
    String(String),
    Boolean(bool)
}

impl Value {
    pub fn integer(value: u64) -> Self {
        Value::Integer(value)
    }

    pub fn float(value: f64) -> Self {
        Value::Float(value)
    }

    pub fn string(value: String) -> Self {
        Value::String(value)
    }

    pub fn boolean(value: bool) -> Self {
        Value::Boolean(value)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Unit => write!(f, "()"),
            Value::Integer(val) => write!(f, "{}", val),
            Value::Float(val) => write!(f, "{}", val),
            Value::String(val) => write!(f, "{}", val),
            Value::Boolean(val) => write!(f, "{}", val),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    Equal,
    NotEqual,
    LogicalAnd,
    LogicalOr,
    LogicalNot,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    BitwiseNot,
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Plus => write!(f, "+"),
            Operator::Minus => write!(f, "-"),
            Operator::Multiply => write!(f, "*"),
            Operator::Modulo => write!(f, "%"),
            Operator::Divide => write!(f, "/"),
            Operator::LessThan => write!(f, "<"),
            Operator::LessThanOrEqual => write!(f, "<="),
            Operator::GreaterThan => write!(f, ">"),
            Operator::GreaterThanOrEqual => write!(f, ">="),
            Operator::Equal => write!(f, "=="),
            Operator::NotEqual => write!(f, "!="),
            Operator::LogicalAnd => write!(f, "&&"),
            Operator::LogicalOr => write!(f, "||"),
            Operator::LogicalNot => write!(f, "!"),
            Operator::BitwiseAnd => write!(f, "&"),
            Operator::BitwiseOr => write!(f, "|"),
            Operator::BitwiseXor => write!(f, "^"),
            Operator::LeftShift => write!(f, "<<"),
            Operator::RightShift => write!(f, ">>"),
            Operator::BitwiseNot => write!(f, "~"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Delimiter {
    ParenthesisLeft,
    ParenthesisRight,
    BracketLeft,
    BracketRight,
    BraceLeft,
    BraceRight
}

impl std::fmt::Display for Delimiter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Delimiter::ParenthesisLeft => write!(f, "("),
            Delimiter::ParenthesisRight => write!(f, ")"),
            Delimiter::BracketLeft => write!(f, "["),
            Delimiter::BracketRight => write!(f, "]"),
            Delimiter::BraceLeft => write!(f, "{{"),
            Delimiter::BraceRight => write!(f, "}}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
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

impl std::fmt::Display for Keyword {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Keyword::If => write!(f, "if"),
            Keyword::Else => write!(f, "else"),
            Keyword::While => write!(f, "while"),
            Keyword::For => write!(f, "for"),
            Keyword::Loop => write!(f, "loop"),
            Keyword::Break => write!(f, "break"),
            Keyword::Continue => write!(f, "continue"),
            Keyword::Let => write!(f, "let"),
            Keyword::Const => write!(f, "const"),
        }
    }
}

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

pub fn expr(input: &str) -> Result<Program, AstError> {
    let mut lexer = Lexer::new(input)?;
    let mut statements = Vec::new();
    loop {
        let statement = expr_bp(&mut lexer, 0)?;
        statements.push(statement);
        let token = lexer.peek();
        let semicolon = token.map(|t| t.get_token());
        match semicolon {
            Some(&LexicalToken::Semicolon) => {
                lexer.next(); // Consume the semicolon
                continue;
            },
            Some(_) => {
                return Err(AstError::UnexpectedToken(token.unwrap().clone(), line!()));
            },
            // Some(&LexicalToken::EndOfFile) => {
            //     break;
            // },
            None => break,
        }
    }

    Ok(Program::new(statements))
}

fn expr_bp(lexer: &mut Lexer, min_bp: u8) -> Result<AstNode, AstError> {
    let current_token = lexer.next();
    let (lhs, current_token) = match current_token {
        Some(token) => match AstNode::atom(token.get_token()) {
            Some(node) => (node, token),
            None => return Err(AstError::UnexpectedToken(token.clone(), line!())),
        },
        None => return Err(AstError::UnexpectedEndOfFile),
    };

    let atom = lhs.get_atom();
    let atom = match atom {
        Some(a) => a,
        None => unreachable!(), // Since lhs is guaranteed to be an atom here
    };

    let mut lhs = match atom.kind() {
        AtomKind::Delimiter => {
            let delimiter = atom.get_delimiter().unwrap();
            match delimiter {
                Delimiter::ParenthesisLeft => {
                    let lhs = expr_bp(lexer, 0)?;
                    if lexer.next().map(|t| t.get_token()) != Some(&LexicalToken::ParenthesisRight) {
                        return Err(AstError::UnmatchedParentheses);
                    }

                    lhs
                },
                _ => {
                    return Err(AstError::UnexpectedToken(current_token.clone(), line!()));
                }
            }
        },
        AtomKind::Operator => {
            let ((), r_bp) = prefix_binding_power(&lhs, &current_token)?;
            let rhs = expr_bp(lexer, r_bp)?;
            AstNode::Expression {
                head: Box::new(lhs),
                args: vec![Box::new(rhs)],
            }
        },
        AtomKind::Keyword => {
            let keyword = atom.get_keyword().unwrap();
            let current_token = current_token.clone();
            match keyword {
                Keyword::If => {
                    let test = expr_bp(lexer, 0)?;
                    if lexer.next().map(|t| t.get_token()) != Some(&LexicalToken::BraceLeft) {
                        return Err(AstError::UnexpectedToken(current_token.clone(), line!()));
                    }

                    let consequent = expr_bp(lexer, 0)?;
                    if lexer.next().map(|t| t.get_token()) != Some(&LexicalToken::BraceRight) {
                        return Err(AstError::UnmatchedBraces);
                    }

                    let next_token = lexer.peek();
                    let alternative = if next_token.map(|t| t.get_token()) == Some(&LexicalToken::Else) {
                        lexer.next(); // Consume 'else'
                        let next_token = lexer.peek().map(|t| t.get_token());
                        
                        let else_if;
                        if next_token == Some(&LexicalToken::BraceLeft)  {
                            lexer.next(); // Consume '{'
                            else_if = false;
                        }
                        else if next_token == Some(&LexicalToken::If) {
                            // Don't consume 'if' here, let the recursive call handle it
                            else_if = true;
                        }
                        else{
                            return Err(AstError::UnexpectedToken(current_token.clone(), line!()));
                        }

                        let alt = expr_bp(lexer, 0)?;
                        // Else if will share the closing brace (and parsing if already checks for matching braces)
                        if lexer.next().map(|t| t.get_token()) != Some(&LexicalToken::BraceRight) && !else_if {
                            return Err(AstError::UnmatchedBraces);
                        }

                        Some(Box::new(alt))
                    }
                    else{
                        None
                    };

                    AstNode::Condition {
                        test: Box::new(test),
                        consequent: Box::new(consequent),
                        alternate: alternative,
                    }
                }
                Keyword::Else => return Err(AstError::UnexpectedToken(current_token.clone(), line!())),
                Keyword::Let => parse_assignment(lexer, false)?,
                Keyword::Const => parse_assignment(lexer, true)?,
                _ => return Err(AstError::UnexpectedEndOfFile), // Placeholder for future keyword handling
            }
        }
        _ => lhs,
    };

    loop {
        let current_token = lexer.peek();
        let (op, current_token) = match current_token {
            Some(token) => match AstNode::atom(token.get_token()) {
                Some(node) if node.get_atom().map(|a| a.is_operator() || a.is_delimiter()).unwrap_or(false) => (node, token),
                //Some(node) if node.get_atom().map(|a| a.is_delimiter()).unwrap_or(false) => break, // 
                Some(_) => return Err(AstError::UnexpectedToken(token.clone(), line!())),
                None => return Err(AstError::UnexpectedToken(token.clone(), line!())),
            },
            None => break, // End of input, break the loop and return lhs
        };

        if let Some((l_bp, ())) = postfix_binding_power(&op, current_token)? {
            if l_bp < min_bp {
                // If the left binding power is less than the minimum binding power, we stop parsing
                break;
            }

            // Consumption of token must occur within the blocks to satisfy lifetime constraints between current_token and lexer
            lhs =  if current_token.get_token() == &LexicalToken::BracketLeft {
                lexer.next(); // Consume the operator
                let rhs = expr_bp(lexer, 0)?;
                if lexer.next().map(|t| t.get_token()) != Some(&LexicalToken::BracketRight) {
                    return Err(AstError::UnmatchedBrackets);
                }

                AstNode::Expression {
                    head: Box::new(op),
                    args: vec![Box::new(lhs), Box::new(rhs)],
                }
            }
            else{
                lexer.next(); // Consume the operator
                AstNode::Expression {
                    head: Box::new(op),
                    args: vec![Box::new(lhs)],
                }
            };

            continue;
        }

        if let Some((l_bp, r_bp)) = infix_binding_power(&op, current_token)?{
            if l_bp < min_bp {
                // If the left binding power is less than the minimum binding power, we stop parsing
                break;
            }

            lexer.next(); // Consume the operator
            let rhs = expr_bp(lexer, r_bp)?;

            lhs = AstNode::Expression {
                head: Box::new(op),
                args: vec![Box::new(lhs), Box::new(rhs)],
            };

            continue;
        }

        break;
    }

    Ok(lhs)
}

fn parse_assignment(lexer: &mut Lexer, constant: bool) -> Result<AstNode, AstError> {
    let identifier_token = lexer.next();
    let identifier_token = match identifier_token {
        Some(token) => token,
        None => return Err(AstError::UnexpectedEndOfFile),
    };

    let identifier = match AstNode::atom(identifier_token.get_token()) {
        Some(node) => match node.get_atom() {
            Some(atom) if atom.is_identifier() => node,
            _ => return Err(AstError::UnexpectedToken(identifier_token.clone(), line!())),
        },
        None => return Err(AstError::UnexpectedToken(identifier_token.clone(), line!())),
    };

    let equal_token = lexer.next();
    let equal_token = match equal_token {
        Some(token) => token,
        None => return Err(AstError::UnexpectedEndOfFile),
    };

    if equal_token.get_token() != &LexicalToken::EqualSign {
        return Err(AstError::UnexpectedToken(equal_token.clone(), line!()));
    }

    let value = expr_bp(lexer, 0)?;

    Ok(AstNode::Assignment { target: Box::new(identifier), value: Box::new(value), constant })
}

fn prefix_binding_power(op: &AstNode, current_token: &LexicalTokenContext) -> Result<((), u8), AstError> {
    match op.get_atom() {
        Some(Atom::Operator(operator)) => match operator {
            Operator::Plus | Operator::Minus => Ok(((), 23)),
            Operator::BitwiseNot | Operator::LogicalNot => Ok(((), 23)),
            _ => Err(AstError::UnexpectedToken(current_token.clone(), line!())),
        },
        _ => Err(AstError::UnexpectedToken(current_token.clone(), line!())),
    }
}

fn postfix_binding_power(op: &AstNode, current_token: &LexicalTokenContext) -> Result<Option<(u8, ())>, AstError> {
    match op.get_atom() {
        Some(Atom::Delimiter(delimiter)) => match delimiter {
            Delimiter::BracketLeft => Ok(Some((25, ()))),
            _ => Ok(None),
        },
        Some(Atom::Operator(_)) => Ok(None),
        _ => Err(AstError::UnexpectedToken(current_token.clone(), line!())),
    }
}

fn infix_binding_power(op: &AstNode, current_token: &LexicalTokenContext) -> Result<Option<(u8, u8)>, AstError> {
    match op.get_atom() {
        Some(Atom::Operator(operator)) => match operator {
            // TODO: Add assignment operator with lowest precedence
            Operator::LogicalOr => Ok(Some((3, 4))),
            Operator::LogicalAnd => Ok(Some((5, 6))),
            Operator::BitwiseOr => Ok(Some((7, 8))),
            Operator::BitwiseXor => Ok(Some((9, 10))),
            Operator::BitwiseAnd => Ok(Some((11, 12))),
            Operator::Equal | Operator::NotEqual => Ok(Some((13, 14))),
            Operator::LessThan | Operator::LessThanOrEqual
            | Operator::GreaterThan | Operator::GreaterThanOrEqual => Ok(Some((15, 16))),
            Operator::LeftShift | Operator::RightShift => Ok(Some((17, 18))),
            Operator::Plus | Operator::Minus => Ok(Some((19, 20))),
            Operator::Multiply | Operator::Divide | Operator::Modulo => Ok(Some((21, 22))),
            _ => Ok(None),
        },
        Some(Atom::Delimiter(_)) => Ok(None),
        _ => Err(AstError::UnexpectedToken(current_token.clone(), line!())),
    }
}

#[cfg(test)]
mod ast_test {
    use super::*;

    #[test]
    fn test_parsing_1() {
        let input = "1 + 2";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(+ 1 2)");
    }

    #[test]
    fn test_parsing_2() {
        let input = "1 + 2 * 3";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(+ 1 (* 2 3))");
    }

    #[test]
    fn test_parsing_3() {
        let input = "--1 * 2";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(* (- (- 1)) 2)");
    }

    #[test]
    fn test_parsing_4() {
        let input = "(((0)))";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "0");
    }

    #[test]
    fn test_parsing_5() {
        let input = "x[0][1]";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "([ ([ x 0) 1)");
    }

    #[test]
    fn test_parsing_6() {
        let input = "if (x < 10) { x + 1 } else { x + 2 }";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(if (< x 10) { (+ x 1) } else { (+ x 2) })");
    }

    #[test]
    fn test_parsing_7() {
        let input = "if (x < 10) { x + 1 }";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(if (< x 10) { (+ x 1) })");
    }

    #[test]
    fn test_parsing_8() { 
        let input = "if x == 0 { 1 } else if x == 1 { 2 } else { 3 }";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        // Else if and else { if } are parsed into the same structure
        assert_eq!(ast.to_string(), "(if (== x 0) { 1 } else { (if (== x 1) { 2 } else { 3 }) })");
    }

    #[test]
    fn test_parsing_9() { 
        let input = "if x == 0 { 1 } else { if x == 1 { 2 } else { 3 } }";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        // Else if and else { if } are parsed into the same structure
        assert_eq!(ast.to_string(), "(if (== x 0) { 1 } else { (if (== x 1) { 2 } else { 3 }) })");
    }

    #[test]
    fn test_parsing_10() { 
        let input = "if x == 0 { 1 } else if x >= 1 && x <= 10 { 2 } else { 3 }";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        // Else if and else { if } are parsed into the same structure
        assert_eq!(ast.to_string(), "(if (== x 0) { 1 } else { (if (&& (>= x 1) (<= x 10)) { 2 } else { 3 }) })");
    }

    #[test]
    fn test_parsing_11() {
        let input = "if (x < 10 || x > 5) { x + 1 }";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(if (|| (< x 10) (> x 5)) { (+ x 1) })");
    }

    #[test]
    fn test_parsing_12() {
        let input = "let x = 10";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(let x = 10)");
    }

    #[test]
    fn test_parsing_13() {
        let input = "((1 + 2) * -(3 + 4)) + 5 * 6 * 7";
        let result = expr(input);
        let ast = &result.unwrap().body[0];
        assert_eq!(ast.to_string(), "(+ (* (+ 1 2) (- (+ 3 4))) (* (* 5 6) 7))");
    }
}