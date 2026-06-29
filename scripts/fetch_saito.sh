#!/usr/bin/env bash
set -euo pipefail

repo_url="https://github.com/SaitoTech/saito-lite-rust.git"
target_dir="third_party/saito-lite-rust"
lock_file="third_party/SAITO_COMMIT"

mkdir -p third_party

if [ -d "$target_dir/.git" ]; then
  git -C "$target_dir" fetch --depth 1 origin master
  git -C "$target_dir" checkout --detach origin/master
else
  git clone --depth 1 "$repo_url" "$target_dir"
fi

git -C "$target_dir" rev-parse HEAD > "$lock_file"
echo "Pinned Saito commit: $(cat "$lock_file")"

