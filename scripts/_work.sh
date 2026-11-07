# Sourced by the other scripts. Makes a scratch copy of the repository in $WORK (default ./work) so that
# reruns never overwrite the shipped results. code/, results/, logs/ are copied; data/ is linked.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-$REPO/work}"
PY="${PY:-python}"
rm -rf "$WORK"; mkdir -p "$WORK" "$REPO/data"
cp -r "$REPO/code" "$REPO/results" "$REPO/logs" "$WORK/"
ln -s "$REPO/data" "$WORK/data"
export PY
