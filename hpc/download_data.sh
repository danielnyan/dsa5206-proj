#!/usr/bin/env bash
# Run on a login/data-transfer node with outbound internet and scratch space.
set -euo pipefail
data_root=${1:?Usage: download_data.sh DATA_ROOT [raw-crowd|normalized]}
variant=${2:-raw-crowd}
case "$variant" in raw-crowd|normalized) ;; *) exit 2;; esac
mkdir -p "$data_root/archives" "$data_root/keras/models"
fetch() {
  local name=$1 expected=$2 target="$data_root/archives/$1"
  if [[ -f "$target" ]] && echo "$expected  $target" | md5sum -c - >/dev/null 2>&1; then return; fi
  curl -fL --retry 8 --retry-all-errors --continue-at - \
    "https://zenodo.org/records/14178894/files/$name?download=1" --output "$target"
  echo "$expected  $target" | md5sum -c -
}
fetch Annotations.zip 7dc40c923f77a251ae564977647938b1
if [[ "$variant" == raw-crowd ]]; then
  fetch Patches.zip dd7c63e595793aad46c07b0f0578e0b1
else
  fetch NormalizedPatches.zip 74ebb5ceed5a8d8a0b0e2df960f7bd94
fi
fetch NormalizedSICAPv2_Annotations.zip 0cbd85834d5ba72cdceb2d76066c1914
fetch NormalizedSICAPv2.zip 61be06796ede2d8c48f01b45b7306a16
weights="$data_root/keras/models/nasnet_large_no_top.h5"
# This is the NASNetLarge no-top MD5 used by Keras applications.
if ! [[ -f "$weights" ]] || ! echo "d81d89dc07e6e56530c4e77faddd61b5  $weights" | md5sum -c - >/dev/null 2>&1; then
  curl -fL --retry 8 --retry-all-errors --continue-at - \
    https://storage.googleapis.com/tensorflow/keras-applications/nasnet/NASNet-large-no-top.h5 \
    --output "$weights"
fi
echo "d81d89dc07e6e56530c4e77faddd61b5  $weights" | md5sum -c -
echo 'Downloads complete. Submit extraction/preprocessing through PBS.'
