// #[macro_use] extern crate prettytable;

pub mod coins;
use coins::Coins;
use std::io::stdin;
// use std::env;

/// Contains the states for the interactive CLI
enum State {
    Menu,
    Load,
    Choice(char),
    SetTarget,
    Compute(i32),
}

/// Gets next state from menu
///
/// q to quit, l to reload coin values, s to set target
///
/// # char -> Returned state -> Next state
/// ```
/// q -> State::Choice('q') -> State::None
///
/// l -> State::Choice('l') -> State::Load
///
/// s -> State::Choice('l') -> State::SetTarget
///
/// _ -> State::Choice(_)   -> State::Menu
///
/// failed parsing -> State::None
/// ```
fn menu() -> Option<State> {
    println!("q to quit, l to reload coin values, s to set target");
    let mut buffer = String::new();
    match stdin().read_line(&mut buffer) {
        Ok(_) => match buffer.trim().chars().next() {
            None => Some(State::Menu),
            Some(c) => Some(State::Choice(c)),
        },
        Err(_) => None,
    }
}

/// Parses input for the amount of change wanted and returns the next state
fn set_target() -> Option<State> {
    println!("enter number to set new target");
    let mut buffer = String::new();
    match stdin().read_line(&mut buffer) {
        Ok(_) => match buffer.trim().parse::<i32>() {
            Ok(v) => Some(State::Compute(v)),
            Err(_) => Some(State::SetTarget),
        }
        Err(_) => None,
    }
}

/// Parses input to select the change making algorithm, launches that algorithm, and returns
/// the next state at the end.
fn with_method(c : &mut Coins, v : i32) -> Option<State> {
    println!("s for successive tries, d for dynamic programming, g for greedy algorithm");
    let mut buffer = String::new();
    match stdin().read_line(&mut buffer) {
        Ok(_) => match buffer.trim().chars().next() {
            None => Some(State::Compute(v)),
            Some(ch) => {
                match ch {
                    's' => coins::min_coins_successive_tries_init(c, v),
                    'd' => coins::min_coins_dynamic_programming(c, v as usize),
                    'g' => coins::min_coins_greedy_algorithm(c, v),
                    _ => (),
                }
                Some(State::Menu)
            }
        }
        Err(_) => None,
    }
}

/// Interactive CLI application
///
/// * Reads the available coin values from config/coins.ron (can be changed and reloaded during runtime)
/// * Parses input to select and execute algorithms
/// * Pretty prints the solution
fn main() {
    // let args: Vec<String> = env::args().collect();
    let input_path = format!("{}/config/coins.ron", env!("CARGO_MANIFEST_DIR"));
    let mut c = Coins::from_file(&input_path);
    let mut state = Some(State::Menu);
    while let Some(s) = state {
        match s {
            State::Menu => state = menu(),
            State::Load => {c = Coins::from_file(&input_path); state = Some(State::Menu)},
            State::Choice(c) => {
                match c {
                    'q' => state = None,
                    'l' => state = Some(State::Load),
                    's' => state = Some(State::SetTarget),
                    _ => state = Some(State::Menu),
                }
            },
            State::SetTarget => state = set_target(),
            State::Compute(v) => {
                c.reset_best();
                state = with_method(&mut c, v);
                c.ppretty_best();
            },
        }
    }
}
