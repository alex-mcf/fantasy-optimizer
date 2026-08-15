#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
archive_commit=5a89742
data_dir="$repo_root/data/raw"

if ! git -C "$repo_root" cat-file -e "$archive_commit^{commit}" 2>/dev/null; then
    echo "Historical commit $archive_commit is unavailable in this clone." >&2
    echo "Fetch full Git history and try again." >&2
    exit 1
fi

mkdir -p "$data_dir"
git -C "$repo_root" archive "$archive_commit" data \
    | tar -x --strip-components=1 -C "$data_dir"

echo "Historical CSVs restored under $data_dir"

