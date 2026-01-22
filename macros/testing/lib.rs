use convert_case::{Case, Casing};
use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::{ExprArray, TypeTuple};

#[derive(deluxe::ParseMetaItem)]
struct ValueParameterizedTestAttribute {
    values: ExprArray,
}

/// ## `value_parameterized_test` Attribute Proc Macro
///
/// The `value_parameterized_test` attribute macro allows you to easily generate multiple test functions from a single test template by specifying a list of values.
/// Each generated test will call the original function with one of the provided values as its argument, making it simple to write parameterized tests.
///
/// ### Features
///
/// - **Parameterized Test Generation:**  
///   Automatically generates a separate test function for each value in the provided array.
/// - **Automatic Naming:**  
///   Each test function is named based on the original function and the value, ensuring unique and descriptive test names.
/// - **Compile-Time Safety:**  
///   Ensures the annotated function has exactly one argument, matching the parameterization.
///
/// ### Example
///
/// ```rust
/// #[value_parameterized_test(values = [1, 2, 3])]
/// fn test_is_positive(x: i32) {
///     assert!(x > 0);
/// }
/// ```
///
/// This will generate three test functions:
/// - `test_is_positive_1`
/// - `test_is_positive_2`
/// - `test_is_positive_3`
///
/// Each will call `test_is_positive` with the corresponding value.
///
/// ### Why use this macro?
///
/// This macro reduces boilerplate and makes it easy to write comprehensive tests for multiple input values, improving code coverage and maintainability.
#[proc_macro_attribute]
pub fn value_parameterized_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the attribute arguments
    let ValueParameterizedTestAttribute { values } =
        deluxe::parse::<ValueParameterizedTestAttribute>(attr).expect("Failed to parse ValueParameterizedTestAttribute");

    let item_fn = syn::parse_macro_input!(item as syn::ItemFn);

    let _item_attrs = &item_fn.attrs;
    let item_vis = &item_fn.vis;
    let item_sig = &item_fn.sig;
    let item_block = &item_fn.block;

    let item_ident = &item_sig.ident;
    let _item_inputs = &item_sig.inputs;

    if item_sig.inputs.len() != 1usize {
        panic!("Function must have exactly one argument");
    }

    let test_defs = values.elems.iter().map(|val| {
        // Remove or replace invalid characters for Rust identifiers
        let mut s = val.to_token_stream().to_string();
        s = s.replace(['"', '\'', '.', '-', '[', ']', '(', ')', '{', '}', ','], "_");
        s = s.replace(' ', "_");

        let test_name = syn::Ident::new(&format!("{}_{}", item_ident, s.to_case(Case::Snake)), item_ident.span());
        quote! {
            #[test]
            fn #test_name() {
                #item_ident(#val)
            }
        }
    });

    quote! {
        #item_vis #item_sig #item_block

        #(#test_defs)*
    }
    .into()
}

#[derive(deluxe::ParseMetaItem)]
struct TypeParameterizedTestAttribute {
    values: TypeTuple,
}

/// ## `type_parameterized_test` Attribute Proc Macro
///
/// The `type_parameterized_test` attribute macro allows you to easily generate multiple test functions from a single generic test template by specifying a list of types.
/// Each generated test will call the original function with one of the provided types as its generic parameter, making it simple to write type-parameterized tests.
///
/// ### Features
///
/// - **Type-Parameterized Test Generation:**  
///   Automatically generates a separate test function for each type in the provided tuple.
/// - **Automatic Naming:**  
///   Each test function is named based on the original function and the type, ensuring unique and descriptive test names.
/// - **Compile-Time Safety:**  
///   Ensures the annotated function has zero arguments and is generic over the specified types.
///
/// ### Example
///
/// ```rust
/// #[type_parameterized_test(values = (u32, i32, f64))]
/// fn test_default<T: Default + std::fmt::Debug>() {
///     let value = T::default();
///     println!("{:?}", value);
/// }
/// ```
///
/// This will generate three test functions:
/// - `test_default_u32`
/// - `test_default_i32`
/// - `test_default_f64`
///
/// Each will call `test_default` with the corresponding type as its generic parameter.
///
/// ### Why use this macro?
///
/// This macro reduces boilerplate and makes it easy to write comprehensive tests for multiple types, improving code coverage and maintainability.
#[proc_macro_attribute]
pub fn type_parameterized_test(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the attribute arguments
    let TypeParameterizedTestAttribute { values } =
        deluxe::parse::<TypeParameterizedTestAttribute>(attr).expect("Failed to parse TypeParameterizedTestAttribute");

    let item_fn = syn::parse_macro_input!(item as syn::ItemFn);

    let _item_attrs = &item_fn.attrs;
    let item_vis = &item_fn.vis;
    let item_sig = &item_fn.sig;
    let item_block = &item_fn.block;

    let item_ident = &item_sig.ident;
    let _item_inputs = &item_sig.inputs;

    if item_sig.inputs.len() != 0 {
        panic!("Function must have exactly zero arguments");
    }

    let test_defs = values.elems.iter().map(|val| {
        let test_name = syn::Ident::new(
            &format!("{}_{}", item_ident, val.to_token_stream().to_string().to_case(Case::Snake)),
            item_ident.span(),
        );
        quote! {
            #[test]
            fn #test_name() {
                #item_ident::<#val>()
            }
        }
    });

    quote! {
        #item_vis #item_sig #item_block

        #(#test_defs)*
    }
    .into()
}
