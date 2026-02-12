# Copters

[![Crates.io](https://img.shields.io/crates/v/copters.svg)](https://crates.io/crates/copters)
[![Documentation](https://docs.rs/copters/badge.svg)](https://docs.rs/copters)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://github.com/amadavan/copters/actions/workflows/rust.yml/badge.svg?branch=main)](https://github.com/amadavan/copters/actions/workflows/rust.yml)

A high-performance Rust library for numerical optimization, providing efficient implementations of algorithms for linear and nonlinear programming.

## Overview

Copters is designed to be a comprehensive optimization toolkit for Rust, offering a wide range of algorithms for solving constrained and unconstrained optimization problems. Built on top of the [`faer`](https://crates.io/crates/faer) linear algebra library, it provides both ease of use and performance.

## Features

### Algorithms

#### Linear Programming
- [ ] **Revised Simplex Method** - Memory-efficient variant of the simplex method
- [ ] **Revised Dual Simplex** - For problems starting with dual feasibility
- [x] **Mehrotra Predictor-Corrector** - Polynomial-time algorithms for large-scale linear programs

#### General Convex Optimization
- [ ] **ADMM** (Alternating Direction Method of Multipliers) - For distributed and constrained convex optimization
- [ ] **Conjugate Gradient** - For quadratic and nonlinear optimization

#### Nonlinear Optimization
- [ ] **Proximal Gradient Descent** - First-order optimization method
- [ ] **Accelerated Gradient Descent** - Nesterov and momentum-based methods
- [ ] **Stochastic Gradient Descent** - For problems with uncertainty (stochastic programs)
- [ ] **Interior Point Method** - For nonlinear programs with constraints

## Design Goals

1. **Performance**: Leverage Rust's zero-cost abstractions and the efficient `faer` linear algebra library
2. **Correctness**: Extensive testing and validation against known optimization problems
3. **Flexibility**: Pluggable objective functions, constraints, and stopping criteria
4. **Usability**: Intuitive API with sensible defaults
5. **Modularity**: Use only the algorithms you need

## Development Status

**This library is in early development.** APIs are subject to change.

Current status:
- [x] Project structure and dependencies
- [x] Basic linear algebra utilities
- [ ] Core algorithm implementations (in progress)
- [ ] Documentation and examples
- [ ] Comprehensive test suite
- [ ] Benchmarks

## Roadmap

- [ ] Support for automatic differentiation
- [ ] Python bindings via PyO3

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development

```bash
# Build the library
cargo build

# Run tests
cargo test

# Generate documentation
cargo doc --open
```

## Related Projects

- [Optimization.jl](https://github.com/SciML/Optimization.jl) - Julia optimization framework
- [scipy.optimize](https://docs.scipy.org/doc/scipy/reference/optimize.html) - Python optimization toolkit
- [COIN-OR](https://www.coin-or.org/) - C++ optimization libraries
- [HiGHS](https://highs.dev/) - Revised simplex method

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Copters in your research, please cite:

```bibtex
@software{copters,
  author = {Avinash Madavan},
  title = {Copters: A Rust Library for Numerical Optimization},
  year = {2026},
  url = {https://github.com/amadavan/copters}
}
```

## Contact

Avinash Madavan - avinash.madavan@gmail.com

Project Link: [https://github.com/amadavan/copters](https://github.com/amadavan/copters)

## AI Disclosure

This project makes limited use of AI-assisted tooling. AI is employed for debugging assistance, boilerplate generation, and drafting documentation. All algorithm design, core implementation, and architectural decisions are made by the author. AI-generated content is reviewed before inclusion.
