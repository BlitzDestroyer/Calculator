use std::collections::HashMap;

use thiserror::Error;

use crate::{lexing::engine::LexicalTokenizeError, parsing::{AstError, AstNode, Atom, Operator, Value}};

#[derive(Debug, Error)]
pub enum ComputationError {
    #[error("Expected value atom")]
    ExpectedValueAtom,
    #[error("Expected operator")]
    ExpectedOperator,
    #[error("Incorrect number of arguments")]
    IncorrectNumberOfArguments,
    #[error("Incompatible types")]
    IncompatibleTypes,
    #[error("AstError: {0}")]
    AstError(#[from] AstError),
    #[error("Expected boolean value")]
    ExpectedBooleanValue,
    #[error("Cannot reassign to constant variable")]
    CannotReassignToConstantVariable,
    #[error("Variable not defined")]
    VariableNotDefined,
    #[error("Not implemented")]
    NotImplemented,
}

#[derive(Debug)]
pub struct ProgramState {
    frames: Vec<ProgramFrame>,
}

impl ProgramState {
    pub fn new() -> Self {
        Self {
            frames: vec![ProgramFrame::new()],
        }
    }

    pub fn get_variable(&self, name: &str) -> Result<Value, ComputationError> {
        for frame in self.frames.iter().rev() {
            if let Some((value, _)) = frame.variables.get(name) {
                return Ok(value.clone());
            }
        }

        Err(ComputationError::VariableNotDefined)
    }
}

#[derive(Debug)]
pub struct ProgramFrame {
    variables: HashMap<String, (Value, bool)>, // bool indicates if the variable is mutable
}

impl ProgramFrame {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
        }
    }
}

pub fn repl() {
    let mut state = ProgramState::new();
    let mut input = String::new();
    loop {
        print!("> ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();
        std::io::stdin().read_line(&mut input).unwrap();
        let user_input = input.as_str().trim();
        match user_input {
            "q" | "quit" => break,
            "h" | "help" => {
                println!("Available commands:");
                println!("  h | help   - Show this help message");
                println!("  q | quit   - Exit the REPL");
            }
            _ => {
                let result = compute_line(user_input, &mut state);
                match result {
                    Ok(value) => println!("Result: {}", value),
                    Err(err) => print_computation_error(err),
                }
                input.clear();
            }
        }
    }
}

pub fn compute_line(line: &str, state: &mut ProgramState) -> Result<Value, ComputationError> {
    let ast = crate::parsing::expr(line)?;
    let value = get_ast_node_value(&ast.body[0], state)?;

    Ok(value)
}

fn get_ast_node_value(node: &AstNode, state: &mut ProgramState) -> Result<Value, ComputationError> {
    match node {
        AstNode::Atom(atom) => get_atom_value(atom, state),
        AstNode::Expression { head, args } => {
            let args = args.iter().map(|arg| get_ast_node_value(arg, state)).collect::<Result<Vec<_>, _>>()?;
            let op = get_ast_node_operator(head)?;
            compute_operation(op, args)
        },
        AstNode::Assignment { target, value } => {
            let value = get_ast_node_value(value, state)?;
            match target.as_ref() {
                AstNode::Atom(Atom::Identifier(name)) => {
                    for frame in state.frames.iter_mut().rev() {
                        if frame.variables.contains_key(name) {
                            let (_, is_constant) = frame.variables.get(name).unwrap();
                            if *is_constant {
                                return Err(ComputationError::CannotReassignToConstantVariable);
                            }
                            frame.variables.insert(name.clone(), (value.clone(), false));
                            return Ok(value);
                        }
                    }
                    return Err(ComputationError::VariableNotDefined);
                }
                _ => Err(ComputationError::ExpectedValueAtom),
            }
        },
        AstNode::Condition { test, consequent, alternate } => compute_condition(test, consequent, alternate, state),
        AstNode::Declaration { target, value, constant } => {
            let value = get_ast_node_value(value, state)?;
            match target.as_ref() {
                AstNode::Atom(Atom::Identifier(name)) => {
                    if state.frames.last_mut().unwrap().variables.contains_key(name) {
                        let (_, is_constant) = state.frames.last_mut().unwrap().variables.get(name).unwrap();
                        if *is_constant {
                            return Err(ComputationError::CannotReassignToConstantVariable);
                        }
                    }
                    state.frames.last_mut().unwrap().variables.insert(name.clone(), (value.clone(), *constant));
                    Ok(value)
                }
                _ => Err(ComputationError::ExpectedValueAtom),
            }
        }
    }
}

fn compute_condition(condition: &AstNode, consequent: &AstNode, alternate: &Option<Box<AstNode>>, state: &mut ProgramState) -> Result<Value, ComputationError> {
    let value = get_ast_node_value(condition, state)?;
    let bool = match value {
        Value::Boolean(b) => b,
        _ => return Err(ComputationError::ExpectedBooleanValue),
    };

    if bool {
        get_ast_node_value(consequent, state)
    }
    else if let Some(alternate) = alternate {
        get_ast_node_value(alternate, state)
    }
    else {
        Ok(Value::Unit)
    }
}

fn compute_operation(op: Operator, args: Vec<Value>) -> Result<Value, ComputationError> {
    match op {
        Operator::Plus => {
            if args.len() == 1 {
                let value = &args[0];
                match value {
                    Value::Integer(v) => Ok(Value::Integer(*v)),
                    Value::Float(v) => Ok(Value::Float(*v)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l + r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l + r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 + r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l + *r as f64)),
                    (Value::String(l), Value::String(r)) => Ok(Value::String(l.clone() + r)),
                    (Value::String(l), Value::Integer(r)) => Ok(Value::String(l.clone() + &r.to_string())),
                    (Value::String(l), Value::Float(r)) => Ok(Value::String(l.clone() + &r.to_string())),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            } 
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::Minus => {
            if args.len() == 1 {
                let value = &args[0];
                match value {
                    Value::Integer(v) => Ok(Value::Integer(-(*v as i64) as u64)),
                    Value::Float(v) => Ok(Value::Float(-*v)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l - r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l - r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 - r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l - *r as f64)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::Multiply => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l * r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l * r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 * r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l * *r as f64)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::Divide => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l / r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l / r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 / r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l / *r as f64)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::Modulo => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l % r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Float(l % r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Float(*l as f64 % r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Float(l % *r as f64)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::LessThan => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l < r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l < r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) < *r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l < (*r as f64))),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::LessThanOrEqual => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l <= r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l <= r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) <= *r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l <= (*r as f64))),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::GreaterThan => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l > r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l > r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) > *r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l > (*r as f64))),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::GreaterThanOrEqual => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l >= r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l >= r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) >= *r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l >= (*r as f64))),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::Equal => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l == r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l == r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) == *r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l == (*r as f64))),
                    (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l == r)),
                    (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l == r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::NotEqual => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Boolean(l != r)),
                    (Value::Float(l), Value::Float(r)) => Ok(Value::Boolean(l != r)),
                    (Value::Integer(l), Value::Float(r)) => Ok(Value::Boolean((*l as f64) != *r)),
                    (Value::Float(l), Value::Integer(r)) => Ok(Value::Boolean(*l != (*r as f64))),
                    (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(l != r)),
                    (Value::String(l), Value::String(r)) => Ok(Value::Boolean(l != r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::LogicalAnd => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(*l && *r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::LogicalOr => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Boolean(l), Value::Boolean(r)) => Ok(Value::Boolean(*l || *r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::LogicalNot => {
            if args.len() == 1 {
                let value = &args[0];
                match value {
                    Value::Boolean(v) => Ok(Value::Boolean(!v)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::BitwiseAnd => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l & r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::BitwiseOr => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l | r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::BitwiseXor => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l ^ r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::LeftShift => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l << r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::RightShift => {
            if args.len() == 2 {
                let left = &args[0];
                let right = &args[1];
                match (left, right) {
                    (Value::Integer(l), Value::Integer(r)) => Ok(Value::Integer(l >> r)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::BitwiseNot => {
            if args.len() == 1 {
                let value = &args[0];
                match value {
                    Value::Integer(v) => Ok(Value::Integer(!v)),
                    _ => Err(ComputationError::IncompatibleTypes),
                }
            }
            else {
                Err(ComputationError::IncorrectNumberOfArguments)
            }
        },
        Operator::Assign => Err(ComputationError::NotImplemented),
    }
}

fn get_ast_node_operator(node: &AstNode) -> Result<Operator, ComputationError> {
    match node {
        AstNode::Atom(atom) => get_atom_operator(atom),
        _ => Err(ComputationError::ExpectedOperator),
    }
}

fn get_atom_operator(atom: &Atom) -> Result<Operator, ComputationError> {
    match atom {
        Atom::Operator(operator) => Ok(*operator),
        _ => Err(ComputationError::ExpectedOperator),
    }
}

fn get_atom_value(atom: &Atom, state: &mut ProgramState) -> Result<Value, ComputationError>{
    match atom {
        Atom::Value(value) => Ok(value.clone()),
        Atom::Identifier(name) => state.get_variable(name),
        _ => Err(ComputationError::ExpectedValueAtom),
    }
}

pub fn print_computation_error(err: ComputationError) {
    match err {
        ComputationError::ExpectedValueAtom => println!("Error: Expected value atom"),
        ComputationError::ExpectedOperator => println!("Error: Expected operator"),
        ComputationError::IncorrectNumberOfArguments => println!("Error: Incorrect number of arguments"),
        ComputationError::IncompatibleTypes => println!("Error: Incompatible types"),
        ComputationError::AstError(ast_error) => print_ast_error(ast_error),
        ComputationError::ExpectedBooleanValue => println!("Error: Conditions require a boolean value"),
        ComputationError::NotImplemented => println!("Error: Not implemented"),
        ComputationError::CannotReassignToConstantVariable => println!("Error: Cannot reassign to constant variable"),
        ComputationError::VariableNotDefined => println!("Error: Variable not defined"),
    }
}

fn print_ast_error(err: AstError) {
    match err {
        AstError::LexicalTokenizeError(lexical_tokenize_error) => {
                        match lexical_tokenize_error {
                            LexicalTokenizeError::ParseIntError(parse_int_error) => println!("Unable to parse integer: {}", parse_int_error),
                            LexicalTokenizeError::ParseFloatError(parse_float_error) => println!("Unable to parse float: {}", parse_float_error),
                            LexicalTokenizeError::UnexpectedCharacter(char, (line, col), _) => println!("Unexpected character '{}' at line {}, column {}", char, line, col),
                            LexicalTokenizeError::UnknownEscapeSequence(char, (line, col), _) => println!("Unknown escape sequence '{}' at line {}, column {}", char, line, col),
                            LexicalTokenizeError::UnterminatedStringLiteral => println!("Unterminated string literal"),
                            LexicalTokenizeError::InvalidBinaryLiteral => println!("Invalid binary literal"),
                            LexicalTokenizeError::InvalidOctalLiteral => println!("Invalid octal literal"),
                            LexicalTokenizeError::InvalidHexadecimalLiteral => println!("Invalid hexadecimal literal"),
                            LexicalTokenizeError::TokenTypeCannotBeNoneWhenBufferIsNotEmpty => println!("Token type cannot be None when buffer is not empty"),
                            LexicalTokenizeError::ExpectedTokenButFoundNone((line, col)) => println!("Expected token, but found none at line {}, column {}", line, col),
                            LexicalTokenizeError::Other(message) => println!("Lexical tokenization error: {}", message),
                        }
            },
        AstError::UnexpectedToken(lexical_token_context, _) => {
                let context = lexical_token_context.get_context();
                println!("Unexpected token '{}' at line {}, column {}", context.get_token_literal(), context.get_line(), context.get_column());
            },
        AstError::ExpectedOperator(lexical_token_context) => {
                let context = lexical_token_context.get_context();
                println!("Expected operator at line {}, column {}", context.get_line(), context.get_column());
            },
        AstError::UnexpectedEndOfFile => {
                println!("Unexpected end of file");
            },
        AstError::UnmatchedParentheses => println!("Unmatched parentheses"),
        AstError::UnmatchedBrackets => println!("Unmatched brackets"),
        AstError::UnmatchedBraces => println!("Unmatched braces"),
        AstError::InvalidAssignmentArity => println!("Invalid assignment arity"),
        AstError::InvalidAssignmentTarget(ast) => println!("Invalid assignment target: {}", ast),
    }
}

#[cfg(test)]
mod computing_test {
    use super::*;

    #[test]
    fn test_atom() {
        let mut state = ProgramState::new();
        let atom = Atom::Value(Value::Integer(42));
        let result = get_atom_value(&atom, &mut state).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_expression_addition() {
        let mut state = ProgramState::new();
        let input = "1 + 1";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(2));
    }

    #[test]
    fn test_expression_subtraction() {
        let mut state = ProgramState::new();
        let input = "2 - 1";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_expression_negation() {
        let mut state = ProgramState::new();
        let input = "-1";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(-1i64 as u64));
    }

    #[test]
    fn test_expression_multiplication() {
        let mut state = ProgramState::new();
        let input = "2 * 3";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(6));
    }

    #[test]
    fn test_expression_division() {
        let mut state = ProgramState::new();
        let input = "6 / 2";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_expression_modulo() {
        let mut state = ProgramState::new();
        let input = "5 % 2";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_expression_bitwise_and() {
        let mut state = ProgramState::new();
        let input = "5 & 3";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_expression_bitwise_or() {
        let mut state = ProgramState::new();
        let input = "5 | 3";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(7));
    }

    #[test]
    fn test_expression_bitwise_xor() {
        let mut state = ProgramState::new();
        let input = "5 ^ 3";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(6));
    }

    #[test]
    fn test_expression_bitwise_not() {
        let mut state = ProgramState::new();
        let input = "~5";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(!5));
    }

    #[test]
    fn test_expression_left_shift() {
        let mut state = ProgramState::new();
        let input = "1 << 2";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(4));
    }

    #[test]
    fn test_expression_right_shift() {
        let mut state = ProgramState::new();
        let input = "4 >> 2";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_expression_complex_1() {
        let mut state = ProgramState::new();
        let input = "(12 + 3 * (8 - 5)) / (7 - 2) + 18 / (3 * (2 + 1)) - 4";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Integer(2));
    }

    #[test]
    fn test_expression_complex_2() {
        let mut state = ProgramState::new();
        let input = "(12 + 3 * (8.0 - 5)) / (7 - 2) + 18 / (3 * (2 + 1)) - 4";
        let result = compute_line(input, &mut state).unwrap();
        assert_eq!(result, Value::Float(2.2));
    }
}