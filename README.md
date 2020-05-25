# enssat-algo

Note : Correct compilation and execution has only been tested on Archlinux.

## Prerequisites
Rust (Linux install using rustup). Rust and rustup are also available on some repos depending on the distro. 
For Windows installation see [here](https://forge.rust-lang.org/infra/other-installation-methods.html).
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Build the project
```
cargo build --release
```

## Run the project
Also compiles if needed. Opens an interactive CLI. The coin values can be modified in the config/coins.ron file and reloaded through
the menu.

```
cargo run --release
```

## Build the doc
Easy documentation navigation through built-in html doc generation. `--open` to automatically open the index in a browser.

```
cargo doc [--open]
```

## Run tests
Runs tests verifying all three algorithms, and the effect of sorting the coins in decreasing order first.

```
cargo test --release
```

To see the output of functions during tests

```
cargo test --release -- --nocapture --test-threads=1
```
