//! Compile-time symbolic automatic differentiation.
//!
//! This crate provides the [`gradient`] and [`hessian`] proc-macro attributes,
//! which parse a Rust function body at compile time, build a symbolic
//! expression tree ([`SymExpr`]), differentiate it analytically, simplify the
//! result, and emit a companion function containing the closed-form derivative.
//!
//! # Supported syntax
//!
//! | Source form | Symbolic node |
//! |---|---|
//! | `x + y`, `x - y`, `x * y`, `x / y` | `Add`, `Sub`, `Mul`, `Div` |
//! | `-x` | `Neg` |
//! | `x.sin()`, `x.cos()` | `Sin`, `Cos` |
//! | `x.ln()`, `x.exp()`, `x.sqrt()` | `Ln`, `Exp`, `Sqrt` |
//! | `x.pow(n)` / `pow(x, n)` (const `n`) | `Pow` |
//! | float literal | `Const` |
//! | function parameter name | `Var(index)` |
//! | `let name = expr;` | inlined into subsequent exprs |
//! | `x as f64`, `(x)` | transparent (inner expr used) |
//! | anything else | `Opaque` (d/dx = 0) |
//!
//! > *AI Disclosure* The structure and partial implementation of this file was generated with AI-tooling.
mod expr;

use expr::*;
use proc_macro2::TokenStream;
use quote::quote;
use std::collections::HashMap;
use syn::{Pat, Stmt};

fn parse_body(block: &syn::Block) -> Option<SymExpr> {
    let mut bindings = HashMap::new();

    for stmt in &block.stmts {
        match stmt {
            Stmt::Local(local) => {
                if let Pat::Ident(pat_ident) = &local.pat {
                    let name = pat_ident.ident.to_string();
                    if let Some(init) = &local.init {
                        let sym = syn_to_sym(&init.expr, &bindings);
                        bindings.insert(name, sym);
                    }
                }
            }
            Stmt::Expr(expr, None) => {
                return Some(syn_to_sym(expr, &bindings));
            }
            _ => {
                // Unsupported statement type (e.g. semi-colon terminated expr, item, macro).
                // For simplicity, we require the function body to be a single expression
                // with optional `let` bindings, so we can skip these.
            }
        }
    }

    None
}

#[derive(deluxe::ParseMetaItem)]
struct DerivativeInput {
    arg: String,
    dim: usize,
    max_passes: Option<usize>,
}

#[proc_macro_attribute]
pub fn gradient(
    attr: proc_macro::TokenStream,
    item: proc_macro::TokenStream,
) -> proc_macro::TokenStream {
    let input_fn = syn::parse_macro_input!(item as syn::ItemFn);
    let fn_name = &input_fn.sig.ident;
    let params = &input_fn.sig.inputs;
    let body = &input_fn.block;
    let vis = &input_fn.vis;

    let DerivativeInput {
        arg,
        dim,
        max_passes,
    } = deluxe::parse::<DerivativeInput>(attr.into())
        .expect("Failed to parse macro attribute arguments for gradient.");

    let sym = match parse_body(body) {
        Some(s) => s,
        None => {
            return syn::Error::new_spanned(
                body,
                "Function body must be a single expression for symbolic differentiation",
            )
            .to_compile_error()
            .into();
        }
    };

    let grad_components: Vec<TokenStream> = (0..dim)
        .map(|i| {
            let d = sym
                .diff(format!("{}[{}]", arg, i))
                .simplify_multipass(max_passes);
            d.into_token_stream()
        })
        .collect();

    let grad_name = syn::Ident::new(
        &format!("{}_gradient", fn_name),
        proc_macro2::Span::call_site(),
    );

    let expanded = quote!(
        #input_fn

        #vis fn #grad_name(#params) -> [f64; #dim] {
            [#(#grad_components),*]
        }
    );

    expanded.into()
}
