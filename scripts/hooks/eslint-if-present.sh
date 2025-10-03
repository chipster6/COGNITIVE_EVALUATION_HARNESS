#!/usr/bin/env bash
set -euo pipefail

# Pass all filenames from pre-commit to eslint if eslint exists; otherwise noop
if ! command -v eslint >/dev/null 2>&1; then
  exit 0
fi

# Collect only JS/TS files from args
files=( )
for f in "$@"; do
  case "$f" in
    *.js|*.jsx|*.ts|*.tsx) files+=("$f") ;;
  esac
done

if [ ${#files[@]} -eq 0 ]; then
  exit 0
fi

# Respect project eslint config; fail on any warning
eslint --max-warnings=0 "${files[@]}"
