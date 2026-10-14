#!/bin/bash
# Weekly upkeep of the llama.cpp fork diskformer runs on (NakliTechie/llama.cpp, branch webgpu-diskformer): merge
# upstream master on a side branch, build, run the paging and MUL_MAT tests, and push the branch only when all pass.
# A conflict or a failure leaves the branch untouched and says so (macOS notification, ntfy when NTFY_TOPIC is set,
# and ~/Library/Logs/diskformer-fork-sync.log). Merges, never rebases, so the push never needs force.
#   bash scripts/sync-llama-fork.sh            (FORK=<checkout> to use another clone)
# The browser build in ggml/dist is not rebuilt here: after a merge, rebuild it and rerun the gates by hand.
set -u
FORK=${FORK:-$HOME/Code/llama.cpp-webgpu-disk}
BR=webgpu-diskformer
LOG=$HOME/Library/Logs/diskformer-fork-sync.log
say() {
  echo "$(date '+%F %H:%M') $1" >> "$LOG"
  osascript -e "display notification \"$1\" with title \"diskformer fork sync\"" >/dev/null 2>&1
  [ -n "${NTFY_TOPIC:-}" ] && curl -s -d "diskformer fork sync: $1" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null
  echo "$1"
}
cd "$FORK" || { say "no fork checkout at $FORK"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] || { say "the fork checkout has uncommitted changes; skipped"; exit 1; }
start=$(git rev-parse --abbrev-ref HEAD)
git fetch -q upstream && git fetch -q origin || { say "fetch failed"; exit 1; }
behind=$(git rev-list --count "origin/$BR..upstream/master")
[ "$behind" = 0 ] && { say "up to date with upstream"; exit 0; }

side=sync-$(date +%F)
git switch -q -C "$side" "origin/$BR"
if ! git merge -q --no-edit upstream/master >> "$LOG" 2>&1; then
  files=$(git diff --name-only --diff-filter=U | tr '\n' ' ')
  git merge --abort; git switch -q "$start"; git branch -q -D "$side"
  say "merging $behind upstream commits conflicts in: $files; needs a hand merge"
  exit 2
fi
back() { git switch -q "$start"; }
if ! cmake --build build-native --target test-webgpu-paged test-backend-ops llama-server -j 8 >> "$LOG" 2>&1; then
  back; say "build failed after merging $behind upstream commits (side branch $side kept)"; exit 3
fi
paged=$(./build-native/bin/test-webgpu-paged 2>&1 | tail -1)
mmid=$(./build-native/bin/test-backend-ops -b WebGPU -o MUL_MAT_ID 2>&1 | grep 'tests passed' | tail -1 | xargs)
mm=$(./build-native/bin/test-backend-ops -b WebGPU -o MUL_MAT 2>&1 | grep 'tests passed' | tail -1 | xargs)
all() { [[ "$1" =~ ^([0-9]+)/([0-9]+) ]] && [ "${BASH_REMATCH[1]}" = "${BASH_REMATCH[2]}" ]; }
if all "$paged" && all "$mmid" && all "$mm"; then
  git switch -q "$BR" && git merge -q --ff-only "$side" && git push -q origin "$BR" && git branch -q -d "$side"
  back
  say "merged $behind upstream commits into $BR and pushed; $paged; MUL_MAT_ID $mmid; MUL_MAT $mm. Rebuild ggml/dist and rerun the gates."
else
  back
  say "tests failed after merging $behind upstream commits (side branch $side kept): paging $paged; MUL_MAT_ID $mmid; MUL_MAT $mm"
  exit 4
fi
