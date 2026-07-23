#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${G_CAMVO_OPTC_SUBSET_MANIFEST:-$ROOT/config/optc_day3_subset_files.tsv}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required" >&2
  exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "OpTC subset manifest not found: $MANIFEST" >&2
  exit 1
fi

download_one() {
  local role="$1"
  local file_id="$2"
  local expected_bytes="$3"
  local relative_destination="$4"
  local destination="$ROOT/$relative_destination"
  local partial="$destination.part"
  local actual_bytes

  mkdir -p "$(dirname "$destination")"
  if [[ -f "$destination" ]]; then
    actual_bytes="$(wc -c < "$destination" | tr -d ' ')"
    if [[ "$actual_bytes" == "$expected_bytes" ]]; then
      echo "verified existing $role file: $relative_destination ($actual_bytes bytes)"
      return
    fi
    echo "existing file has wrong size; moving it back to resumable .part" >&2
    mv "$destination" "$partial"
  fi

  echo "downloading $role file: $relative_destination"
  curl \
    --location \
    --fail \
    --retry 20 \
    --retry-delay 5 \
    --continue-at - \
    --output "$partial" \
    "https://drive.usercontent.google.com/download?id=${file_id}&export=download&confirm=t"

  actual_bytes="$(wc -c < "$partial" | tr -d ' ')"
  if [[ "$actual_bytes" != "$expected_bytes" ]]; then
    echo "size mismatch for $relative_destination: $actual_bytes != $expected_bytes" >&2
    exit 1
  fi
  mv "$partial" "$destination"
  echo "verified $role file: $relative_destination ($actual_bytes bytes)"
}

while IFS=$'\t' read -r role file_id expected_bytes destination; do
  [[ -z "${role:-}" || "$role" == \#* ]] && continue
  download_one "$role" "$file_id" "$expected_bytes" "$destination"
done < "$MANIFEST"

echo "OpTC Day-3 bounded subset download complete."
echo "Attack shards: data/raw/optc-day3/attack"
echo "Benign controls: data/raw/optc-benign"
