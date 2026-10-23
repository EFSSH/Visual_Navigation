#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
VM="$ROOT/../nav-environment/scripts/nav-vm"
"$VM" start -y nav-jazzy
for mode in deps build test; do
  "$VM" shell --workdir=/home/efssh nav-jazzy -- bash /mnt/mac/第一版导航框架/scripts/workspace.sh "$mode"
done
printf '\nBuild and regression tests completed.\n'
