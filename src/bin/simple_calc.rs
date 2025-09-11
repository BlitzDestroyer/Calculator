use calculator::computing::{compute_line, print_computation_error, repl, ProgramState};
use clap::Parser;

#[derive(Parser, Debug)]
#[command(version = "v1.0.0", about = "Simple calculator repl", long_about = r#"Simple 4 function calculator program"#)]
struct Args {
    #[arg(help ="Expression to evaluate")]
    expr: Option<String>,
}

fn main() {
    let args = Args::parse();
    let mut state = ProgramState::new();
    if let Some(expr) = args.expr {
        let result = compute_line(&expr, &mut state);
        match result {
            Ok(value) => println!("Result: {}", value),
            Err(err) => print_computation_error(err),
        }
    }
    else {
        repl();
    }
}