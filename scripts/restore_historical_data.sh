#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_bin="$repo_root/.venv/bin/python"
if [ ! -x "$python_bin" ]; then
    python_bin=python3
fi

"$python_bin" "$repo_root/scripts/import_nflverse_results.py" \
    2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025
"$python_bin" "$repo_root/scripts/import_ffc_adp.py" \
    2018 2019 2020 2021 2022 2023 2024 2025 2026
"$python_bin" "$repo_root/scripts/import_nflverse_players.py"
"$python_bin" "$repo_root/scripts/validate_data.py"

echo "Consistent historical archive refreshed under $repo_root/data/raw"
