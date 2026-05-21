#!/usr/bin/env bash
# Prepend DD-compatible header to each Morgan shard in library_morgan/.
# Phase-1 molecular_file_count_updated.py skips the first line of each file.
#
# Usage: bash utilities/add_morgan_headers.sh /path/to/library_morgan
#
set -euo pipefail

MORGAN_DIR="${1:?Usage: add_morgan_headers.sh MORGAN_DIR}"

for f in "${MORGAN_DIR}"/*.txt; do
  [[ -f "$f" ]] || continue
  if head -n1 "$f" | grep -q '^ZINC_ID,'; then
    echo "skip (has header): $(basename "$f")"
    continue
  fi
  tmp="${f}.header_tmp"
  { echo "ZINC_ID,morgan_bits"; cat "$f"; } > "$tmp"
  mv "$tmp" "$f"
  echo "header added: $(basename "$f")"
done

echo "Finished."
