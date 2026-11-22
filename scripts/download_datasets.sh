#!/usr/bin/env bash
# Download the raw CWRU (12 kHz submitted set and 48 kHz camera-ready set) and Paderborn files
# used by VibeTwin into data/raw/.
# Resumable: finished files are skipped, partial downloads continue. Paderborn
# downloads retry every RETRY_SECONDS until the university server answers.
set -u
cd "$(dirname "$0")/.."
RETRY_SECONDS="${RETRY_SECONDS:-600}"
SEVENZIP="${SEVENZIP:-data/raw/_tools/7zz}"

echo "[cwru] start $(date -Is)"
# Layout follows data/metadata/cwru/manifest.csv: data/raw/cwru/<group>/<file>.mat
tail -n +2 data/metadata/cwru/manifest.csv | while IFS=, read -r rel fname _cls _load _cond src _rest; do
  [ -s "$rel" ] && continue
  mkdir -p "$(dirname "$rel")"
  curl -fsSL --retry 5 -o "$rel.part" "https://engineering.case.edu/sites/default/files/${src}.mat" \
    && mv "$rel.part" "$rel" && echo "[cwru] $fname" || echo "[cwru] FAILED $fname"
done

echo "[cwru48k] start $(date -Is)"
# Camera-ready CWRU set, every file sampled at 48 kHz (data/metadata/cwru48k/file_map.csv):
# data/raw/cwru48k/<subset_group>/<file>.mat
tail -n +2 data/metadata/cwru48k/file_map.csv | while IFS=, read -r fname _cls _load _cond src group; do
  rel="data/raw/cwru48k/$group/$fname"
  [ -s "$rel" ] && continue
  mkdir -p "$(dirname "$rel")"
  curl -fsSL --retry 5 -o "$rel.part" "https://engineering.case.edu/sites/default/files/${src}.mat" \
    && mv "$rel.part" "$rel" && echo "[cwru48k] $fname" || echo "[cwru48k] FAILED $fname"
done

BEARINGS="K001 K002 K003 K004 K005 K006 KA01 KA03 KA04 KA05 KA06 KA07 KA08 KA09 KA15 KA16 KA22 KA30 KB23 KB24 KB27 KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21"
ARCH=data/raw/_archives/paderborn
mkdir -p "$ARCH" data/raw/paderborn
while :; do
  pending=0
  for b in $BEARINGS; do
    [ -f "$ARCH/$b.done" ] && continue
    curl -fsSL -C - --retry 3 -o "$ARCH/$b.rar" "https://groups.uni-paderborn.de/kat/BearingDataCenter/$b.rar"
    got=$?
    if [ -s "$ARCH/$b.rar" ] && "$SEVENZIP" t "$ARCH/$b.rar" >/dev/null 2>&1; then
      "$SEVENZIP" x -y -aos -o"data/raw/paderborn" "$ARCH/$b.rar" >/dev/null \
        && touch "$ARCH/$b.done" && echo "[paderborn] $b ok $(date -Is)" && continue
      echo "[paderborn] $b extract failed"; pending=1
    elif [ "$got" = 0 ]; then
      echo "[paderborn] $b archive corrupt, re-downloading"; rm -f "$ARCH/$b.rar"; pending=1
    else
      echo "[paderborn] $b not available yet (curl $got) $(date -Is)"; pending=1
      [ -s "$ARCH/$b.rar" ] || rm -f "$ARCH/$b.rar"
    fi
  done
  [ "$pending" = 0 ] && break
  sleep "$RETRY_SECONDS"
done
echo "[all] done $(date -Is)"
