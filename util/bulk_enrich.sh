#!/usr/bin/env bash
# Bulk enrichment driver: runs 11 batches of 1000 rows sequentially.
# Idempotent — picks rows WHERE address_fetched_at IS NULL.
set -u
cd "$(dirname "$0")/.."

PY=".venv/Scripts/python.exe"
BATCH_LIMIT=1000
export PYTHONPATH=.

echo "=== BULK ENRICHMENT START at $(date -u +%H:%M:%SZ) ==="
for i in 2 3 4 5 6 7 8 9 10 11 12; do
  echo "=== batch $i starting at $(date -u +%H:%M:%SZ) ==="
  "$PY" util/enrich_addresses.py --limit "$BATCH_LIMIT" 2>&1 | grep --line-buffered -vE "^geocoding:|it/s\\]" || true
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "=== batch $i FAILED rc=$rc at $(date -u +%H:%M:%SZ) ==="
  else
    echo "=== batch $i ended at $(date -u +%H:%M:%SZ) ==="
  fi
done
echo "=== ALL BATCHES COMPLETE at $(date -u +%H:%M:%SZ) ==="
