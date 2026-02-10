#![feature(map_try_insert)]
#![allow(static_mut_refs)]

use std::sync::Mutex;

use convert_case::{Case, Casing};
use lazy_static::lazy_static;
use proc_macro::TokenStream;
use quote::{ToTokens, quote};
use syn::punctuated::Punctuated;
use syn::{Expr, Ident, LitStr, MetaNameValue, Token, TraitBound, Type, TypeTuple};

/// ## `explicit_options` Attribute Proc Macro
///
/// The `explicit_options` attribute macro is used to annotate a struct, enabling explicit
/// registration and management of solver options. It processes all fields marked with the
/// `#[use_option(...)]` attribute, collecting their metadata (name, type, default value,
/// description) and generating an internal options struct for type-safe access.
///
/// ### Features
///
/// - **Explicit Option Registration:**   Only fields explicitly marked with `#[use_option(...)]`
///   are registered as solver options.
/// - **Internal Options Struct Generation:**   Generates an internal struct containing all
///   registered options, allowing type-safe access and conversion from a global options registry.
/// - **Compile-Time Validation:**   Ensures that option names and types are consistent and prevents
///   duplicate or conflicting registrations.
/// - **Integration with Option Registry:**   Works with other macros in the crate to provide
///   dynamic documentation and runtime option management.
///
/// ### Example
///
/// ```rust
/// #[explicit_options]
/// pub struct MyOptions {
///     #[use_option(name = "tolerance", type_ = f64, default = "1e-8", description = "Convergence tolerance")]
///     tolerance: f64,
///     // other fields...
/// }
/// ```
///
/// This will register the `tolerance` field as a solver option and generate an internal options
/// struct for type-safe access.
///
/// ### Why use an attribute macro?
///
/// Using an attribute macro allows you to explicitly control which fields are registered as
/// options, keeping your code clear and maintainable. It also enables automatic documentation and
/// validation, reducing manual bookkeeping and errors.
#[proc_macro_attribute]
pub fn explicit_options(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let item_struct = syn::parse_macro_input!(item as syn::ItemStruct);
    let struct_attrs = &item_struct.attrs;
    let vis = &item_struct.vis;
    let ident = &item_struct.ident;
    let generics = &item_struct.generics;
    let struct_token = &item_struct.struct_token;

    let item_attr = struct_attrs
        .iter()
        .filter(|attr| attr.path().is_ident("use_option"));

    let option_set: Vec<(LitStr, Type)> = item_attr
        .clone()
        .map(|attr| {
            let name_values: Result<Punctuated<MetaNameValue, Token![,]>, _> =
                attr.parse_args_with(Punctuated::parse_terminated);
            let mut name: Option<LitStr> = None;
            let mut type_: Option<Type> = None;
            if let Ok(name_values) = name_values {
                for nv in name_values {
                    if nv.path.get_ident().is_none() {
                        panic!("Expected a name-value meta item");
                    }

                    let option_name = nv.path.get_ident().unwrap();
                    match option_name.to_string().as_str() {
                        "name" => {
                            // Handle name
                            name = Some(
                                syn::parse2::<LitStr>(nv.value.clone().into_token_stream())
                                    .expect("Failed to parse name"),
                            );
                        }
                        "type_" => {
                            // Handle type
                            type_ = Some(
                                syn::parse2::<Type>(nv.value.clone().into_token_stream())
                                    .expect("Failed to parse type"),
                            );
                        }
                        "default" => {
                            // Handle default value
                        }
                        "description" => {
                            // Handle description
                        }
                        _ => (),
                    }
                }
            } else {
                panic!("Expected a name-value meta item");
            }
            let name = name.expect("Option 'name' is required");
            let type_ = type_.expect("Option 'type_' is required");
            (name, type_)
        })
        .collect();

    let option_vals = option_set
        .iter()
        .map(|(name, type_)| {
            let name_expr = syn::parse_str::<Expr>(&format!("{}", name.value()))
                .expect("Failed to parse name as Expr");
            quote! {
                #name_expr: #type_
            }
        })
        .collect::<Vec<_>>();

    let option_def = option_set.iter().map(|(name, type_)| {
        let name_expr = syn::parse_str::<Expr>(&format!("{}", name.value()))
            .expect("Failed to parse name as Expr");
        quote! {
            #name_expr: options.get_option::<#type_>(#name).expect("Option not found").clone()
        }
    });

    let internal_options_ident = syn::Ident::new(
        &format!("{}InternalOptions", ident),
        proc_macro2::Span::call_site(),
    );

    let fields: Vec<proc_macro2::TokenStream> = item_struct
        .fields
        .iter()
        .map(|f| {
            let field_name = &f.ident;
            let field_type = &f.ty;
            quote! {
                #field_name: #field_type
            }
        })
        .collect();

    let expanded = quote! {
        #[derive(Debug, Clone)]
        pub(crate) struct #internal_options_ident {
            #(#option_vals),*
        }

        impl std::convert::From<&crate::Options> for #internal_options_ident {
            fn from(options: &crate::Options) -> Self {
                #internal_options_ident {
                    #(#option_def),*
                }
            }
        }

        #(#struct_attrs)*
        #vis #struct_token #ident #generics {
            options: #internal_options_ident,
            #(#fields),*
        }
    };

    expanded.into()
}

#[derive(deluxe::ParseMetaItem)]
struct OptionInput {
    name: LitStr,
    type_: Type,
    default: Option<LitStr>,
    description: Option<LitStr>,
}

lazy_static! {
    static ref OptionMap: Mutex<Option<std::collections::HashMap<String, Box<(String, Option<String>, Option<String>)>>>> =
        Mutex::new(Some(Default::default()));
}

/// ## `use_option` Attribute Proc Macro
///
/// The `use_option` attribute macro is used to annotate struct fields, registering them as solver
/// options with associated metadata. When applied, it records the field's name, type, default
/// value, and description in the global option registry for use in code generation and
/// documentation.
///
/// ### Features
///
/// - **Option Registration:**   Automatically registers the annotated field as an available solver
///   option in the crate-wide registry.
/// - **Metadata Storage:**   Stores the option's name, type, default value, and description for use
///   in documentation and code generation.
/// - **Integration with Other Macros:**   Enables dynamic documentation and registry generation by
///   macros like `build_options!`, `gen_option_struct!`, and `explicit_options`.
///
/// ### Example
///
/// ```rust
/// #[use_option(name = "tolerance", type_ = f64, default = "1e-8", description = "Convergence tolerance")]
/// pub struct MyOptions {
///     tolerance: f64,
///     // other fields...
/// }
/// ```
///
/// This will register the `tolerance` option with its metadata, making it available for dynamic
/// documentation and option management.
///
/// ### Why use an attribute macro?
///
/// Using an attribute macro allows you to annotate options directly in your struct definitions,
/// keeping option registration and metadata close to the code. This reduces manual bookkeeping and
/// ensures documentation and registries are always up to date.
#[proc_macro_attribute]
pub fn use_option(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the attribute arguments
    let OptionInput {
        name,
        type_,
        default,
        description,
    } = deluxe::parse::<OptionInput>(attr).expect("Failed to parse OptionInput");

    let type_ident = &type_;
    let _default_lit = &default;

    if default.is_some() {
        OptionMap
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .try_insert(
                name.value(),
                Box::new((
                    type_ident.to_token_stream().to_string(),
                    Some(default.to_token_stream().to_string()),
                    Some(description.as_ref().map_or("".to_string(), |d| d.value())),
                )),
            )
            .map_err(|mut err| {
                if err.value.0 != err.entry.get().0 {
                    panic!(
                        "Option '{}' is already defined with a different type",
                        name.value()
                    );
                }

                if err.entry.get().1.is_some()
                    && default.is_some()
                    && err.entry.get().1.as_ref().unwrap() != &default.to_token_stream().to_string()
                {
                    panic!(
                        "Option '{}' is already defined with a different default value",
                        name.value()
                    );
                }

                *err.entry.get_mut() = Box::new((
                    err.entry.get().0.clone(),
                    Some(default.to_token_stream().to_string()),
                    Some(description.as_ref().map_or("".to_string(), |d| d.value())),
                ));
            })
            .ok();
    } else {
        OptionMap
            .lock()
            .unwrap()
            .as_mut()
            .unwrap()
            .try_insert(
                name.value(),
                Box::new((type_ident.to_token_stream().to_string(), None, None)),
            )
            .map_err(|err| {
                if err.value.0 != err.entry.get().0 {
                    panic!(
                        "Option '{}' is already defined with a different type",
                        name.value()
                    );
                }
            })
            .ok();
    }

    let item_struct = syn::parse_macro_input!(item as syn::ItemStruct);

    // Generate code to register the option at runtime before the function body
    // let struct_block = &item_struct.block;
    // let struct_attrs = &item_struct.attrs;
    // let struct_vis = &item_struct.vis;
    // let struct_sig = &item_struct.sig;

    quote! (
        #item_struct
    )
    .into()
}

#[derive(deluxe::ParseMetaItem)]
struct OptionBuilder {
    registry_name: Expr,
}
/// ## `build_options!` Proc Macro
///
/// The `build_options!` macro generates a static registry containing all solver options defined in
/// your crate. It collects metadata for each registered option—including name, type, default value,
/// and description—and produces both the registry and auto-generated documentation.
///
/// ### Features
///
/// - **Static Registry Generation:**   Creates a static variable (e.g., `OPTION_REGISTRY`) that
///   maps option names to their values and types for use at runtime.
/// - **Auto-Generated Documentation:**   Dynamically generates a Markdown table listing each
///   option's name, type, default value, and description, ensuring documentation always matches the
///   actual options available in your code.
/// - **Dynamic Doc Generation:**   As a proc macro, it inspects the registered options and
///   generates up-to-date documentation at compile time.
///
/// ### Example
///
/// ```rust
/// build_options!(name = OPTION_REGISTRY);
/// ```
///
/// This will generate a static registry named `OPTION_REGISTRY` with documentation that includes a
/// table of all available options.
///
/// ### Why use a proc macro?
///
/// Using a proc macro allows the registry and its documentation to be generated dynamically, so it
/// always matches the options defined in your codebase. This reduces manual maintenance and helps
/// keep your documentation accurate and complete.
#[proc_macro]
pub fn build_options(tokens: TokenStream) -> TokenStream {
    let OptionBuilder { registry_name } =
        deluxe::parse::<OptionBuilder>(tokens).expect("Failed to parse OptionBuilder");

    let option_map_guard = OptionMap.lock().unwrap();
    let option_map = option_map_guard.as_ref().unwrap();

    let options_fields: Vec<_> = option_map
        .iter()
        .map(|(key, value)| {
            let (type_str, default, _description) = value.as_ref();
            let type_ident: Type = syn::parse_str(type_str).expect("Failed to parse type");
            let default = default
                .as_ref()
                .unwrap_or(&"Default::default()".to_string())
                .to_string()
                .replace("\"", "");
            quote! {
                (#key.to_string(), Box::new(#default.parse::<#type_ident>().expect("Failed to parse default value")) as Box<dyn crate::OptionTrait>)
            }
        })
        .collect();

    let docs_fields: Vec<_> = option_map
        .iter()
        .map(|(key, value)| {
            let (type_str, default, description) = value.as_ref();
            let _type_: Type = syn::parse_str(type_str).expect("Failed to parse type");
            let default = default
                .as_ref()
                .unwrap_or(&"Default::default()".to_string())
                .to_string()
                .replace("\"", "");
            let description = description
                .as_ref()
                .unwrap_or(&"".to_string())
                .to_string()
                .replace("\"", "");
            format!(
                "| {} | [`{}`] | {} | {} |",
                key,
                type_str.replace(" :: ", "::"),
                default,
                description
            )
        })
        .collect();

    let mut doc_string = String::from(
        "Option registry for Options.\n\n| Option Name      | Type   | Default | Description                \
         |\n|------------------|--------|---------|----------------------------|\n",
    );

    for field in docs_fields {
        doc_string.push_str(&format!("{}\n", field));
    }

    let registry_name = registry_name
        .to_token_stream()
        .to_string()
        .replace("\"", "");
    let registry_ident = syn::Ident::new(&registry_name, proc_macro2::Span::call_site());

    let expanded = quote! {
        #[doc = #doc_string]
        static #registry_ident : std::sync::LazyLock<std::collections::HashMap<String, Box<dyn crate::OptionTrait>>> = std::sync::LazyLock::new(|| {
            let mut map : std::collections::HashMap::<String, Box<dyn crate::OptionTrait>> = std::collections::HashMap::new();
            map.extend([#(#options_fields),*]);
            map
        });

        #[doc = #doc_string]
        #[derive(Clone)]
        pub struct Options {
            map: std::collections::HashMap<String, Box<dyn crate::OptionTrait>>,
        }

        impl Options {
            pub fn new() -> Self {
                let map = #registry_ident.clone();
                Self { map }
            }

            pub fn get_option<T: OptionTrait>(&self, name: &str) -> Option<T>
            where
                T: Clone,
            {
                self.map
                    .get(name)
                    .and_then(|v| {
                        // Downcast to the concrete type
                        (v.as_ref() as &dyn Any).downcast_ref::<T>()
                    })
                    .cloned()
            }

            pub fn set_option<T: OptionTrait>(&mut self, name: &str, value: T) -> Result<(), String> {
                if !self.map.contains_key(name) {
                    return Err(format!("Option '{}' is not registered.", name));
                }

                if let Some(_) = (self.map.get(name).unwrap().as_ref() as &dyn Any).downcast_ref::<T>() {
                    self.map.insert(name.to_string(), Box::new(value));
                    Ok(())
                } else {
                    Err(format!(
                        "Type mismatch for option '{}'. Expected {}, found {}.",
                        name,
                        std::any::type_name::<T>(),
                        "unknown type",
                    ))
                }
            }
        }
    };

    expanded.into()
}

#[derive(deluxe::ParseMetaItem)]
struct EnumTraitInput {
    trait_: TraitBound,
    name: LitStr,
    variants: TypeTuple,
    new_arguments: TypeTuple,
    doc_header: Option<LitStr>,
}

/// ## `build_option_enum!` Proc Macro
///
/// The `build_option_enum!` macro generates an enum type that acts as a registry for all available
/// implementations of a trait (such as initializers or solvers).
///
/// ### Features
///
/// - **Enum Generation:**   Automatically creates an enum (e.g., `Initializers`) with variants for
///   each registered implementation.
/// - **Auto-Generated Documentation:**   Dynamically generates documentation for the enum,
///   including a Markdown table listing each variant, its type, default value, and description.
/// - **Dynamic Doc Generation:**   As a proc macro, it inspects the registered implementations and
///   generates up-to-date documentation at compile time.
/// - **Variant Construction:**   The generated enum includes methods for constructing trait objects
///   from enum variants and for listing all available variants.
///
/// ### Example
///
/// ```rust
/// build_option_enum!(
///     trait_ = Initializer,
///     name = "Initializers",
///     variants = (SimpleInitializer, AdvancedInitializer),
///     new_arguments = (),
///     doc_header = "Initializer registry"
/// );
/// ```
///
/// This will generate an `Initializers` enum with documentation that includes a table of all
/// available variants and methods for trait object construction.
///
/// ### Why use a proc macro?
///
/// Using a proc macro allows the documentation and enum registry to be generated dynamically, so it
/// always matches the implementations defined in your codebase. This reduces manual maintenance and
/// helps keep your documentation accurate and complete.
#[proc_macro]
pub fn build_option_enum(token: TokenStream) -> TokenStream {
    let EnumTraitInput {
        trait_,
        name,
        variants,
        new_arguments,
        doc_header,
    } = deluxe::parse::<EnumTraitInput>(token).expect("Failed to parse EnumTraitInput");

    let enum_name = syn::Ident::new(&name.value(), name.span());

    let variant_types: Vec<_> = variants.elems.iter().map(|v| v.to_token_stream()).collect();
    let variant_names: Vec<_> = variants
        .elems
        .iter()
        .map(|v| {
            syn::parse_str::<Ident>(&v.to_token_stream().to_string().to_case(Case::Snake)).unwrap()
        })
        .collect();

    let argument_types: Vec<_> = new_arguments
        .elems
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let arg_ident = syn::Ident::new(&format!("arg{}", i), proc_macro2::Span::call_site());
            quote! { #arg_ident: #v }
        })
        .collect();
    let argument_idents: Vec<_> = (0..new_arguments.elems.len())
        .map(|i| syn::Ident::new(&format!("arg{}", i), proc_macro2::Span::call_site()))
        .collect();
    let argument_ident = quote!(
        #(#argument_idents),*
    );

    let doc_header = quote!(#doc_header)
        .to_string()
        .trim_matches('"')
        .to_string()
        + " The ```Default::default``` values for the enum is ```"
        + &variant_names[0].to_string()
        + "```.";

    let expanded = quote! {
        use std::str::FromStr;

        #[derive(Clone, Debug, Default)]
        #[doc = #doc_header]
        pub enum #enum_name {
            #[default]
            #(#variant_names),*
        }

        impl crate::OptionTrait for #enum_name {}

        impl FromStr for #enum_name {
            type Err = &'static str;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    #(#variant_names => Ok(#enum_name::#variant_names),)*
                    _ => Err("Invalid enum variant"),
                }
            }
        }

        impl #enum_name {
            pub const variants: &[#enum_name] = &[ #(#enum_name::#variant_names),* ];

            pub fn into_variant(type_ : #enum_name, #(#argument_types),*) -> Box<dyn #trait_> {
                match type_ {
                    #(#variant_names => Box::new(#variant_types::new(#argument_ident)),)*
                }
            }
        }
    };

    expanded.into()
}
