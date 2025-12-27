#!/bin/sh
# Wrapper to inject common flags for mojo commands
# Usage: ./scripts/mojo_wrap.sh <command> <file> [args...]

CMD=$1
shift
mojo "$CMD" -D STIMOJO_INT_TYPE="$STIMOJO_INT_TYPE" -I src "$@"
