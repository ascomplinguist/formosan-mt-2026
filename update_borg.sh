#!/usr/bin/env bash
set -euo pipefail

BORG=borg
REPO=kevin-ckpts
BASE="work"
NEW_ARCHIVE="kv20260922"

# Collect top-level folders already present in any archive.
existing=$(
    "$BORG" list --short "$REPO" |
    while read -r archive; do
        "$BORG" list --short "$REPO::$archive"
    done |
    sed -n "s#^${BASE}/\([^/]*\)/.*#\1#p" |
    sort -u
)

# Find current folders that are not already archived.
missing=()

for dir in "$BASE"/*/; do
    [ -d "$dir" ] || continue

    name=$(basename "$dir")

    if ! grep -Fxq "$name" <<< "$existing" && [ -f "$dir/config.json" ]; then
        missing+=("${dir%/}")
    fi
done

if [ "${#missing[@]}" -eq 0 ]; then
    echo "Nothing new to archive."
    exit 0
fi

echo "Adding to $REPO::$NEW_ARCHIVE:"
printf '  %s\n' "${missing[@]}"

"$BORG" create \
    --stats \
    --progress \
    "$REPO::$NEW_ARCHIVE" \
    "${missing[@]}"
