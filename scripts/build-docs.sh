#!/bin/bash
# Build documentation and copy to docs/

cargo doc "$@"
rm -rf docs/rustdocs
cp -r target/doc docs/rustdocs
