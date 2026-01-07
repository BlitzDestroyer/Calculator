# Calculator

A small CLI calculator written in **Rust**, built around a custom **lexer**, **Pratt-style parser**, and **interpreter**.  
The project is intended both as a usable calculator and as an exploration of language implementation techniques.

---

## Overview

The calculator accepts expressions as input, parses them into an abstract syntax tree (AST), and evaluates the result.  
It supports variables, scoped declarations, conditionals, and a wide range of operators.

The implementation avoids parser generators and instead uses a hand-written lexer and Pratt parser for clarity and extensibility.

At the moment, only four-function artimetic is supported for computations. Parsing can still handle a wide range of expression types.

---

## Usage

### One-shot evaluation
```bash
cargo run -- "1 + 2 * 3"
```

Output:
```text
Result: 7
```

### REPL mode
```bash
cargo run
```

Example session:
```text
> 2 + 3
Result: 5
> 2 + (3 - (4 + 3 + 2)) / (23 - 3)
Result: 1.7
> const 1 + 2 + 3 - 4 * 9
Result: -30
```

---

## Supported Features

### Values
- Integers
- Floating-point numbers
- Booleans
- Strings
- Unit value (`()`)

### Operators

**Arithmetic**
- `+`, `-`, `*`, `/`, `%`
- Unary `-`

**Comparison**
- `<`, `<=`, `>`, `>=`
- `==`, `!=`

**Logical**
- `&&`, `||`, `!`

**Bitwise**
- `&`, `|`, `^`, `~`
- `<<`, `>>`

---

## Language Constructs

### Declarations
```text
let x = 10
const y = 5
```

### Assignments
```text
x = x + 1
```

### Conditionals
```text
if x < 10 { x + 1 } else { x + 2 }
```

### Nested expressions
```text
((1 + 2) * -(3 + 4)) + 5 * 6 * 7
```

---

## Architecture

```text
Input
  ↓
Lexer
  ↓
Parser (Pratt)
  ↓
AST
  ↓
Interpreter
  ↓
Value
```

Each stage is isolated and testable, allowing future extensions such as optimization passes or compilation.

---

## AST Structure

```text
Program
 └── Vec<AstNode>

AstNode
 ├── Atom
 ├── Declaration { target, value, constant }
 ├── Assignment { target, value }
 ├── Expression { head, args }
 └── Condition { test, consequent, alternate }
```

This structure cleanly separates syntax from evaluation semantics.

---

## Scoping and State

- Lexical scoping implemented with a stack of frames
- Shadowing is allowed
- `const` bindings cannot be reassigned
- Variable lookup proceeds from innermost to outermost scope

---

## Error Handling

The interpreter reports detailed errors, including:
- Lexical errors (invalid characters, malformed literals)
- Parse errors (unexpected tokens, unmatched delimiters)
- Runtime errors (type mismatches, invalid assignments)

Errors are surfaced with line and column information when available.

---

## Tests

```bash
cargo test
```

The test suite covers:
- Parsing correctness
- Operator precedence
- Complex expression evaluation
- Runtime error cases

---

## Future Work

- Intermediate representation (SSA-based)
- LaTeX-style syntax support (e.g. `\frac{x}{3}`)
- User-defined functions
- Additional numeric types
- Optional compilation or optimization passes

