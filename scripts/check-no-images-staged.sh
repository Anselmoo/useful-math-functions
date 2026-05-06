#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'
# Enable globstar so allowlist patterns with ** match recursively
shopt -s globstar

# Get staged files (Added, Copied, Modified)
staged_files=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$staged_files" ]; then
  exit 0
fi

disallowed=()

for f in $staged_files; do
  case "$f" in
    *.png|*.PNG|*.gif|*.GIF)
      allowed=false
      if [ -f .image-allowlist ]; then
        while IFS= read -r pattern; do
          pattern=$(echo "$pattern" | sed 's/#.*//; s/^[ \t]*//; s/[ \t]*$//')
          [ -z "$pattern" ] && continue
          if [[ "$f" == $pattern ]]; then allowed=true; break; fi
        done < .image-allowlist
      else
        # default allowlist
        if [[ "$f" == docs/** ]]; then allowed=true; fi
      fi
      if ! $allowed; then disallowed+=("$f"); fi
      ;;
  esac
done

if [ ${#disallowed[@]} -gt 0 ]; then
  echo "Disallowed image files staged (png/gif) outside allowlist:"
  for d in "${disallowed[@]}"; do echo "  $d"; done
  echo "Either remove them from the commit, or add an allowlist entry to .image-allowlist"
  exit 1
fi
exit 0
