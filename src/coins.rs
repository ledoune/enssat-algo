//! Contains the Coins logic and associated algorithms

use ron::de::from_reader;
use serde::Deserialize;
use std::{fs::File,
    fmt,
    thread,
    cmp::Ordering};
use std::time::Instant;


/// Stores all the values needed for the algorithms, and the best solution found if any.
#[derive(Debug, Deserialize, Default, Clone)]
#[serde(default)]
pub struct Coins {
    /// Available coins
    values : Vec<i32>,
    /// Smallest coin
    min_val : Option<i32>,
    /// Biggest con
    max_val : Option<i32>,
    /// Current change
    change : Vec<i32>,
    /// Current change value
    sum : i32,
    /// Number of coins in current change
    nb_coins : i32,
    /// Best solution found
    best : Option<Vec<i32>>,
    /// Number of coins in the best solution found
    nb_coins_best : Option<i32>
}

/// Default display for [Coins](struct.Coins.html)
impl fmt::Display for Coins {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "{:?}\n{:?}\n{:?}\n{:?}\n{:?}", self.values, self.change, self.sum, self.best, self.nb_coins_best)
    }
}

#[allow(dead_code)]
impl Coins {
    /// Constructor for an empty [Coins](struct.Coins.html) instance.
    pub fn new() -> Self {
        Coins {
            values : Vec::new(),
            min_val : None,
            max_val : None,
            change : Vec::new(),
            sum : 0,
            nb_coins : 0,
            best : None,
            nb_coins_best : None,
        }
    }

    /// Contructor using a vector of coins available.
    pub fn from_values(values : Vec<i32>) -> Self {
        let min_val = values.iter().min().map(|v| *v);
        let max_val = values.iter().max().map(|v| *v);
        let change = vec![0; values.len()];
        Coins {
            values,
            min_val,
            max_val,
            change,
            sum : 0,
            nb_coins : 0,
            best : None,
            nb_coins_best : None,
        }
    }

    /// Constructor using a specified path to read available coin values from RON file (pretty prints the values read).
    /// Does not filter duplicate values. In case of read failure exits the program.
    pub fn from_file(path : &str) -> Self {
        // read the RON file for coin values
        let f = File::open(&path).expect("Failed opening file");
        // catch errors
        let mut c : Coins = match from_reader(f) {
            Ok(x) => x,
            Err(e) => {
                println!("Failed to load coins config: {}", e);
                std::process::exit(1);
            }
        };
        c.values.sort_by(|a, b| b.cmp(a));
        // fill the change vector with zeroes
        for _ in 0..c.values.len() {
            c.change.push(0);
        };
        // find the min coin value if there are any
        c.min_val = c.values.iter().min().map(|v| *v);
        c.max_val = c.values.iter().max().map(|v| *v);
        c.ppretty_coins();
        c
    }

    /// Resets best solution found and returns self.
    pub fn reset_best(&mut self) -> &mut Self {
        self.best = None;
        self.nb_coins_best = None;
        self
    }

    /// Returns a clone of the Vector of available values.
    pub fn clone_values(&self) -> Vec<i32> {
        self.values.clone()
    }

    /// Pretty prints the best solution found.
    ///
    /// # Example output
    /// ```
    /// +--------+-----+-----+-----+----+----+----+---+---+---+-----+
    /// | Coins  | 500 | 200 | 100 | 50 | 20 | 10 | 5 | 2 | 1 | 888 |
    /// +--------+-----+-----+-----+----+----+----+---+---+---+-----+
    /// | Change | 0   | 0   | 0   | 1  | 2  | 0  | 1 | 2 | 0 | 6   |
    /// +--------+-----+-----+-----+----+----+----+---+---+---+-----+
    /// | Total  | 0   | 0   | 0   | 50 | 40 | 0  | 5 | 4 | 0 | 99  |
    /// +--------+-----+-----+-----+----+----+----+---+---+---+-----+
    /// ```
    pub fn ppretty_best(&self) -> () {
        use std::str::FromStr;
        use prettytable::{Table, Row, Cell};

        let mut values: Vec<String> = self.values.clone().iter().map(|v| v.to_string()).collect();
        values.insert(0, String::from_str("Coins").unwrap());
        values.push(self.values.iter().sum::<i32>().to_string());
        let values = Row::new(values.iter().map(|s| Cell::new(s)).collect());

        let mut change: Vec<String> = self.best.clone().unwrap_or(vec![]).iter().map(|c| c.to_string()).collect();
        change.insert(0, String::from_str("Change").unwrap());
        change.push(self.nb_coins_best.unwrap_or(0).to_string());
        let change = Row::new(change.iter().map(|s| Cell::new(s)).collect());

        let mut totals: Vec<i32> = self.best.clone().unwrap_or(vec![]).iter().enumerate().map(|(i,c)| self.values[i] * c).collect();
        totals.push(totals.iter().sum());
        let mut totals: Vec<String> = totals.iter().map(|t| t.to_string()).collect();
        totals.insert(0, String::from_str("Total").unwrap());
        let totals = Row::new(totals.iter().map(|s| Cell::new(s)).collect());

        let mut table = Table::new();
        table.add_row(values);
        table.add_row(change);
        table.add_row(totals);
        table.printstd();
    }

    /// Pretty prints the available coins.
    ///
    /// # Example output
    /// ```
    /// +-------+-----+-----+-----+----+----+----+---+---+---+
    /// | Coins | 500 | 200 | 100 | 50 | 20 | 10 | 5 | 2 | 1 |
    /// +-------+-----+-----+-----+----+----+----+---+---+---+
    /// ```
    pub fn ppretty_coins(&self) -> () {
        use std::str::FromStr;
        use prettytable::{Table, Row, Cell};

        let mut values: Vec<String> = self.values.clone().iter().map(|v| v.to_string()).collect();
        values.insert(0, String::from_str("Coins").unwrap());
        let values = Row::new(values.iter().map(|s| Cell::new(s)).collect());

        let mut table = Table::new();
        table.add_row(values);
        table.printstd();
    }

    /// Reset current change to all 0.
    fn reset_change(&mut self) -> &mut Self {
        for v in self.change.iter_mut() {
            *v = 0;
        }
        self
    }

    /// Returns true if
    ///
    /// * The target is superior or equal to the smallest coin available
    /// * Current amount of coins is smaller than in the best solution
    /// * (number of coins in best solution - current number of coins) biggest coins is bigger than target change
    /// * Or target equals 0.
    fn is_viable(&self, target : i32) -> bool {
        match self.min_val {
            None => false,
            Some(v) => match self.nb_coins_best {
                None => target >= v || target == 0,
                Some(nb) => (target >= v && self.nb_coins < nb && target < (nb - self.nb_coins)*self.max_val.unwrap()) || target == 0,
            }
        }
    }

    /// Returns true if the current amount of coins is smaller than the one for the best solution.
    fn is_better(&self) -> bool {
        match self.nb_coins_best {
            None => true,
            Some(nb) => nb > self.nb_coins,
        }
    }

    /// Returns true if target equals 0
    fn is_solution(&self, target : i32) -> bool {
        target == 0
    }
}

// pub fn min_coins<'a>(c : &'a mut Coins, target : i32, mut coins_values_clone : &'a[i32]) -> () {
    // if coins_values_clone.is_empty() {
        // coins_values_clone = c.values.as_mut_slice();
    // }
    // // println!("{:?}", c);
    // if c.is_solution(target) && c.is_better() {
        // c.best = Some(c.change.clone()); c.nb_coins_best = Some(c.nb_coins);
    // } else {
        // for (i,val) in coins_values_clone.iter().enumerate() {
            // c.change[i] += 1; c.sum += val; c.nb_coins += 1;
            // if c.is_viable(target - val) {min_coins(c, target - val, coins_values_clone)};
            // c.change[i] -= 1; c.sum -= val; c.nb_coins -= 1;
        // }
    // }
// }

/// Unsafe variable for multithreaded successive tries
static mut THREAD_RESULTS : Vec<Coins> = Vec::<Coins>::new();

/// Spawns threads for each available coin, join the threads, and sort [results](static.THREAD_RESULTS.html).
fn spawn_threads(target : i32) {
    unsafe {
        let mut children = vec![];
        // Each thread starts with a different coin
        // N.B : in some cases the computation may be considerably slower than a single threaded execution, due to some pruning
        // conditions.
        for (i, c) in THREAD_RESULTS.iter_mut().enumerate() {
            let cloned_values = c.values.clone();
            c.change[i] += 1; c.sum += c.values[i]; c.nb_coins += 1;
            children.push(thread::spawn(move || {
                min_coins_successive_tries(c, target - c.values[i], cloned_values.as_slice());
            }));
        }

        for child in children {
            child.join().expect("Couldn't join on the associated thread");
        }

        // Sort the solutions
        THREAD_RESULTS.sort_by(|a, b| {
            if let Some(best_a) = a.nb_coins_best {
                if let Some(best_b) = b.nb_coins_best {
                    best_a.cmp(&best_b)
                } else {
                    Ordering::Less
                }
            } else {
                Ordering::Greater
            }
        });
    }
}

/// Initiate the successive tries and time the execution. Modifies c to store the best solution found.
/// Does not reset current change.
pub fn min_coins_successive_tries_init(c: &mut Coins, target: i32) {
    // split into threads
    // unsafe because I don't really have time to bother writing idiomatic "safe" Rust code
    println!("Selected successive tries...");
    let tic = Instant::now();
    unsafe {
        for _ in c.values.iter() {
            THREAD_RESULTS.push(c.clone());
        }
        spawn_threads(target);
        if let Some(v) = THREAD_RESULTS.first() {
            c.best = v.best.clone(); c.nb_coins_best = v.nb_coins_best;
        }
        THREAD_RESULTS.clear();
    }
    let toc = tic.elapsed().as_millis();
    println!("Time elapsed: {}ms", toc);
}

/// Initiate the successive tries and time the execution. Modifies c to store the best solution found.
/// Does not reset current change. Single thread version.
#[allow(dead_code)]
pub fn min_coins_successive_tries_single_thread(c: &mut Coins, target: i32) {
    println!("Selected successive tries singled thread...");
    let tic = Instant::now();
    min_coins_successive_tries(c, target, c.values.clone().as_slice());
    let toc = tic.elapsed().as_millis();
    println!("Time elapsed: {}ms", toc);
}

/// For pruning effect test. Recursion only stops if current amount of coins is greater than for the best solution found,
/// or if by adding coin, target is exceeded
#[allow(dead_code)]
fn min_coins_successive_tries_no_pruning(c: &mut Coins, target : i32, values_clone : &[i32]) {
    if c.is_solution(target) && c.is_better() {
        c.best = Some(c.change.clone()); c.nb_coins_best = Some(c.nb_coins);
    } else {
        for (i, val) in values_clone.iter().enumerate() {
            c.change[i] += 1; c.sum += val; c.nb_coins += 1;
            if (target - val >= 0) && (c.nb_coins < c.nb_coins_best.unwrap_or(c.nb_coins + 1)) {min_coins_successive_tries_no_pruning(c, target - val, values_clone)};
            c.change[i] -= 1; c.sum -= val; c.nb_coins -= 1;
        }
    }
}

/// Recursive function implementing successive tries algorithm.
fn min_coins_successive_tries(c : &mut Coins, target : i32, values_clone : &[i32]) {
    // println!("{}", c);
    if c.is_solution(target) && c.is_better() {
        c.best = Some(c.change.clone()); c.nb_coins_best = Some(c.nb_coins);
        // println!("{}", c);
    } else {
        for (i,val) in values_clone.iter().enumerate() {
            c.change[i] += 1; c.sum += val; c.nb_coins += 1;
            if c.is_viable(target - val) {min_coins_successive_tries(c, target - val, values_clone)};
            c.change[i] -= 1; c.sum -= val; c.nb_coins -= 1;
        }
    }
}


/// Holds a solution and its size for the dynamic programming algorithm.
/// See function [min_coins_dynamic_programming](fn.min_coins_dynamic_programming.html).
#[derive(Debug, Default, Clone)]
struct Solution {
    sol : Option<(i32, Vec<i32>)>,
    len : Option<usize>,
}

#[allow(dead_code)]
impl Solution {
    /// Constructor for empty [Solution](struct.Solution.html)
    fn new(size: usize) -> Self {
        Solution {
            sol : Some((0, vec![0; size])),
            len : Some(size),
        }
    }

    /// Constructor for [Solution](struct.Solution.html) from a Vector
    fn from_vec(v: Vec<i32>) -> Self {
        let len = Some(v.len());
        Solution {
            sol : Some((v.iter().sum(), v)),
            len,
        }
    }

    /// Increments the amount of coint at the given index by 1. Does nothing in case of index out of bounds.
    fn inc(&mut self, index: usize) {
        if index < self.len.unwrap_or(index) {
            if let Some((ref mut s, ref mut c)) = self.sol {
                *s += 1;
                c[index] += 1;
            }
        }
    }
}

/// Holds the intermediate solutions for the dynamic programming algorithm.
/// See function [min_coins_dynamic_programming](fn.min_coins_dynamic_programming.html).
#[derive(Debug)]
struct Matrix {
    col : usize,
    row : usize,
    data : Vec<Solution>,
}

#[allow(dead_code)]
impl Matrix {
    /// Contructor for [Matrix](struct.Matrix.html) initiating all values to Option::None.
    fn new(col: usize, row: usize) -> Self {
        Matrix {
            col,
            row,
            data : vec![Solution::default(); col * row],
        }
    }

    /// Contructor for [Matrix](struct.Matrix.html) initiating all values to init_value.
    fn from_init_value(col: usize, row: usize, init_value: Solution) -> Self {
        Matrix {
            col,
            row,
            data : vec![init_value; col * row],
        }
    }

    /// Getter. Panics in case of index out of bounds.
    fn get(&self, col: usize, row: usize) -> &Solution {
        self.data.get(col + self.col * row).unwrap()
    }

    /// Setter. Returns the solution or Result::Err("Index out of bounds").
    fn set(&mut self, col: usize, row: usize, sol: Solution) -> Result<&Solution, &str> {
        if col < self.col && row < self.row {
            self.data[col + self.col * row] = sol;
            Ok(self.get(col, row))
        } else {
            Err("Index out of bounds")
        }
    }

    /// Removes value (replaces by None). Returns Result::Ok(()) or Result::Err("Index out of bounds").
    fn rm(&mut self, col: usize, row: usize) -> Result<(), &str>{
        if col < self.col && row < self.row {
            self.data[col + self.col * row] = Solution::default();
            Ok(())
        } else {
            Err("Index out of bounds")
        }
    }
}

/// Solves the change making problem using dynamic programming and times the execution.
/// # Algorithm
/// * Initiate the matrix with empty solutions.
/// * Remove solutions for 0 coin and change > 0.
/// * For each coin, find best solutions for up to target change using solutions from previous coins.
/// * Memorize the best solution in c.
pub fn min_coins_dynamic_programming(c: &mut Coins, target: usize) {
    println!("Selected dynamic programming...");
    let tic = Instant::now();
    let base_change = Solution::from_vec(vec![0; c.values.len()]);
    let mut change_making_matrix = Matrix::new(c.values.len() + 1, target + 1);
    for i in 1..=target {change_making_matrix.rm(0, i).unwrap()}
    // let base_change = c.values.clone();
    for (i, coin) in c.values.iter().enumerate() {
        let coin = *coin as usize;
        for change in 1..=target {
            if coin == change {
                let mut tmp = base_change.clone(); tmp.inc(i);
                match change_making_matrix.set(i + 1, change, tmp) {
                    Err(e) => println!("{}", e),
                    _ => (),
                };
            } else if coin > change {
                match change_making_matrix.set(i + 1, change, change_making_matrix.get(i, change).clone()) {
                    Err(e) => println!("{}", e),
                    _ => (),
                };
            } else {
                let tmp1 = change_making_matrix.get(i, change).clone();
                let mut tmp2 = change_making_matrix.get(i + 1, change - coin).clone(); tmp2.inc(i);
                let next_sol = match tmp1.sol {
                    None => tmp2,
                    Some((s1, _)) => {
                        if let Some((s2, _)) = tmp2.sol {
                            if s1 < s2 {tmp1} else {tmp2}
                        } else {tmp1}
                    },
                };
                if let Err(e) = change_making_matrix.set(i + 1, change, next_sol) {
                    println!("{}", e);
                }
            }
        }
    }

    if let Some((nb_coins_best, best)) = change_making_matrix.get(c.values.len(), target).clone().sol {
        c.best = Some(best); c.nb_coins_best = Some(nb_coins_best);
    }

    let toc = tic.elapsed().as_micros();
    println!("Time elaped: {}us", toc);
}

/// Assuming coin values are already sorded in decreasing order, computes a solution by adding the biggest coin possible to the
/// change until the target is reached
pub fn min_coins_greedy_algorithm(c: &mut Coins, target: i32) {
    let mut target = target as i64;
    println!("Selected greedy algorithm...");
    let tic = Instant::now();
    for (i, coin) in c.values.iter().enumerate() {
        let coin = *coin as i64;
        while target - coin >= 0 {
            target -= coin;
            c.change[i] += 1;
        }
        if target == 0 {break;}
    }
    c.best = Some(c.change.clone());
    c.nb_coins_best = Some(c.change.iter().sum());

    let toc = tic.elapsed().as_nanos();
    println!("Time elapsed: {}ns", toc);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_coins() -> Coins {
        let values = vec![50, 20, 10, 5, 2, 1];
        Coins::from_values(values)
    }

    fn target() -> i32 {
        99
    }

    fn solution() -> Option<Vec<i32>> {
        Some(vec![1, 2, 0, 1, 2, 0])
    }

    #[test]
    // Unit test for successive tries
    fn succ_tries() {
        let mut c = init_coins();
        min_coins_successive_tries_single_thread(&mut c, target());
        assert_eq!(c.best, solution());
    }

    #[test]
    // Unit test for dynamic prog
    fn dyn_prog() {
        let mut c = init_coins();
        min_coins_dynamic_programming(&mut c, target() as usize);
        assert_eq!(c.best, solution());
    }

    #[test]
    // Unit test for greedy algo
    fn greedy() {
        let mut c = init_coins();
        min_coins_greedy_algorithm(&mut c, target());
        assert_eq!(c.best, solution());
    }

    #[test]
    // Empirical test for the effect of sorting the coins in descending orders.
    fn pruning_sorting() {
        let mut c = init_coins();
        for t in 55..=target() {
            c.reset_best(); c.values.reverse();
            let tic_increasing = Instant::now();
            min_coins_successive_tries_single_thread(&mut c, t);
            let toc_increasing = tic_increasing.elapsed();

            c.reset_best(); c.values.reverse();
            let tic_decreasing = Instant::now();
            min_coins_successive_tries_single_thread(&mut c, t);
            let toc_decreasing = tic_decreasing.elapsed();

            assert!(toc_increasing > toc_decreasing);
        }
    }

    #[test]
    // Empirical test for the effect of pruning
    fn pruning_target() {
        let mut c = init_coins();
        // target needs to be superior to a certain amount for pruning to be efficient
        // due to the cost of some test in is_viable()
        for t in 300..=333 {
            c.reset_best();
            let tic_pruning = Instant::now();
            min_coins_successive_tries_single_thread(&mut c, t);
            let toc_pruning = tic_pruning.elapsed();

            c.reset_best();
            let v = c.values.clone();
            let tic_no_pruning = Instant::now();
            min_coins_successive_tries_no_pruning(&mut c, t, v.as_slice());
            let toc_no_pruning = tic_no_pruning.elapsed();

            eprintln!("Pruning: {}ms - No pruning {}ms", toc_pruning.as_millis(), toc_no_pruning.as_millis());
            assert!(toc_no_pruning >= toc_pruning);
        }
    }
}
