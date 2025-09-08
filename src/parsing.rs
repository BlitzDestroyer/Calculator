use thiserror::Error;

use crate::lexing::{Lexer, LexicalToken, LexicalTokenContext, LexicalTokenizeError};

#[derive(Debug)]
pub enum AstNode {
    Atom(Atom),
    Assignment { target: Box<AstNode>, value: Box<AstNode> },
    Expression { head: Box<AstNode>, args: Vec<Box<AstNode>> },
    Condition { test: Box<AstNode>, consequent: Box<AstNode>, alternate: Box<AstNode> },
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

    fn get_condition(&self) -> Option<(&AstNode, &AstNode, &AstNode)> {
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
            AstNode::Assignment { target, value } => write!(f, "({} = {})", target, value),
            AstNode::Expression { head, args } => {
                let args_str: Vec<String> = args.iter().map(|arg| format!("{}", arg)).collect();
                write!(f, "({} {})", head, args_str.join(" "))
            }
            AstNode::Condition { test, consequent, alternate } => write!(f, "(if ({}) {{{}}} else {{{}}})", test, consequent, alternate),
        }
    }
}

#[derive(Debug)]
pub enum Atom {
    Value(Value),
    Operator(Operator),
    Identifier(String),
}

impl Atom {
    fn value(lexical_token: &LexicalToken) -> Option<Atom> {
        let val = match lexical_token {
            LexicalToken::Integer(value) => Some(Value::Integer(*value)),
            LexicalToken::Float(value) => Some(Value::Float(*value)),
            LexicalToken::StringLiteral(value) => Some(Value::String(value.clone())),
            LexicalToken::Boolean(value) => Some(Value::Bool(*value)),
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
            LexicalToken::LessThanSign => Some(Operator::LessThan),
            LexicalToken::LessThanEqualSign => Some(Operator::LessThanOrEqual),
            LexicalToken::GreaterThanSign => Some(Operator::GreaterThan),
            LexicalToken::GreaterThanEqualSign => Some(Operator::GreaterThanOrEqual),
            LexicalToken::EqualEqualSign => Some(Operator::Equal),
            LexicalToken::ExclamationMarkEqualSign => Some(Operator::NotEqual),
            LexicalToken::Ampersand => Some(Operator::BitwiseAnd),
            LexicalToken::Pipe => Some(Operator::BitwiseOr),
            LexicalToken::Caret => Some(Operator::BitwiseXor),
            LexicalToken::LessThanLessThanSign => Some(Operator::LeftShift),
            LexicalToken::GreaterThanGreaterThanSign => Some(Operator::RightShift),
            LexicalToken::ExclamationMark => Some(Operator::ExclamationMark),
            LexicalToken::ParenthesisLeft => Some(Operator::ParenthesisLeft),
            LexicalToken::ParenthesisRight => Some(Operator::ParenthesisRight),
            LexicalToken::BracketLeft => Some(Operator::BracketLeft),
            LexicalToken::BracketRight => Some(Operator::BracketRight),
            LexicalToken::BraceLeft => Some(Operator::BraceLeft),
            LexicalToken::BraceRight => Some(Operator::BraceRight),
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

    fn is_value(&self) -> bool {
        matches!(self, Atom::Value(_))
    }

    fn is_operator(&self) -> bool {
        matches!(self, Atom::Operator(_))
    }

    fn is_identifier(&self) -> bool {
        matches!(self, Atom::Identifier(_))
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
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atom::Value(value) => write!(f, "{}", value),
            Atom::Operator(operator) => write!(f, "{}", operator),
            Atom::Identifier(name) => write!(f, "{}", name),
        }
    }
}

#[derive(Debug)]
pub enum Value {
    Integer(u64),
    Float(f64),
    String(String),
    Bool(bool)
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Integer(val) => write!(f, "{}", val),
            Value::Float(val) => write!(f, "{}", val),
            Value::String(val) => write!(f, "{}", val),
            Value::Bool(val) => write!(f, "{}", val),
        }
    }
}

#[derive(Debug)]
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
    And,
    Or,
    ExclamationMark,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    LeftShift,
    RightShift,
    Negate,
    ParenthesisLeft,
    ParenthesisRight,
    BracketLeft,
    BracketRight,
    BraceLeft,
    BraceRight
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
            Operator::And => write!(f, "&&"),
            Operator::Or => write!(f, "||"),
            Operator::ExclamationMark => write!(f, "!"),
            Operator::BitwiseAnd => write!(f, "&"),
            Operator::BitwiseOr => write!(f, "|"),
            Operator::BitwiseXor => write!(f, "^"),
            Operator::LeftShift => write!(f, "<<"),
            Operator::RightShift => write!(f, ">>"),
            Operator::Negate => write!(f, "~"),
            Operator::ParenthesisLeft => write!(f, "("),
            Operator::ParenthesisRight => write!(f, ")"),
            Operator::BracketLeft => write!(f, "["),
            Operator::BracketRight => write!(f, "]"),
            Operator::BraceLeft => write!(f, "{{"),
            Operator::BraceRight => write!(f, "}}"),
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
    #[error("Unexpected end of file")]
    UnexpectedEndOfFile,
}

pub fn expr(input: &str) -> Result<AstNode, AstError> {
    let mut lexer = Lexer::new(input)?;
    expr_bp(&mut lexer, 0)
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
        None => return Err(AstError::UnexpectedToken(current_token.clone(), line!())),
    };

    let mut lhs = if atom.is_operator() {
        let op = lhs.get_atom().unwrap().get_operator().unwrap();
        match op {
            Operator::ParenthesisLeft => {
                let lhs = expr_bp(lexer, 0)?;
                assert_eq!(lexer.next().map(|t| t.get_token()), Some(&LexicalToken::ParenthesisRight));
                lhs
            },
            _ => {
                let ((), r_bp) = prefix_binding_power(&lhs, &current_token)?;
                let rhs = expr_bp(lexer, r_bp)?;
                AstNode::Expression {
                    head: Box::new(lhs),
                    args: vec![Box::new(rhs)],
                }
            }
        }
    }
    else{
        lhs
    };

    loop {
        let current_token = lexer.peek();
        let (op, current_token) = match current_token {
            Some(token) => match AstNode::atom(token.get_token()) {
                Some(node) if node.get_atom().map(|a| a.is_operator()).unwrap_or(false) => (node, token),
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
                assert_eq!(lexer.next().map(|t| t.get_token()), Some(&LexicalToken::BracketRight));
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

fn prefix_binding_power(op: &AstNode, current_token: &LexicalTokenContext) -> Result<((), u8), AstError> {
    match op.get_atom() {
        Some(Atom::Operator(operator)) => match operator {
            Operator::Plus | Operator::Minus => Ok(((), 17)),
            Operator::Negate | Operator::ExclamationMark => Ok(((), 17)),
            _ => Err(AstError::UnexpectedToken(current_token.clone(), line!())),
        },
        _ => Err(AstError::UnexpectedToken(current_token.clone(), line!())),
    }
}

fn postfix_binding_power(op: &AstNode, current_token: &LexicalTokenContext) -> Result<Option<(u8, ())>, AstError> {
    match op.get_atom() {
        Some(Atom::Operator(operator)) => match operator {
            Operator::BracketLeft => Ok(Some((19, ()))),
            _ => Ok(None),
        },
        _ => Err(AstError::UnexpectedToken(current_token.clone(), line!())),
    }
}

fn infix_binding_power(op: &AstNode, current_token: &LexicalTokenContext) -> Result<Option<(u8, u8)>, AstError> {
    match op.get_atom() {
        Some(Atom::Operator(operator)) => match operator {
            Operator::BitwiseOr => Ok(Some((1, 2))),
            Operator::BitwiseXor => Ok(Some((3, 4))),
            Operator::BitwiseAnd => Ok(Some((5, 6))),
            Operator::Equal | Operator::NotEqual => Ok(Some((7, 8))),
            Operator::LessThan | Operator::LessThanOrEqual | Operator::GreaterThan | Operator::GreaterThanOrEqual => Ok(Some((9, 10))),
            Operator::LeftShift | Operator::RightShift => Ok(Some((11, 12))),
            Operator::Plus | Operator::Minus => Ok(Some((13, 14))),
            Operator::Multiply | Operator::Divide | Operator::Modulo => Ok(Some((15, 16))),
            _ => Ok(None),
        },
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
        let ast = result.unwrap();
        assert_eq!(ast.to_string(), "(+ 1 2)");
    }

    #[test]
    fn test_parsing_2() {
        let input = "1 + 2 * 3";
        let result = expr(input);
        let ast = result.unwrap();
        assert_eq!(ast.to_string(), "(+ 1 (* 2 3))");
    }

    #[test]
    fn test_parsing_3() {
        let input = "--1 * 2";
        let result = expr(input);
        let ast = result.unwrap();
        assert_eq!(ast.to_string(), "(* (- (- 1)) 2)");
    }

    #[test]
    fn test_parsing_4() {
        let input = "(((0)))";
        let result = expr(input);
        let ast = result.unwrap();
        assert_eq!(ast.to_string(), "0");
    }

    #[test]
    fn test_parsing_5() {
        let input = "x[0][1]";
        let result = expr(input);
        let ast = result.unwrap();
        assert_eq!(ast.to_string(), "([ ([ x 0) 1)");
    }
}