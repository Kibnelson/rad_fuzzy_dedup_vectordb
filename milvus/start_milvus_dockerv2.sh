#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# Resolve RAD_WS + MILVUS_DIR
#   - default: infer from this script's location
#   - optional: --ws /path/to/rad_workspace
# ------------------------------------------------------------
WS_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ws)
      WS_OVERRIDE="${2:-}"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      echo "Usage: $0 [--ws /abs/path/to/rad_workspace]" >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "$WS_OVERRIDE" ]]; then
  RAD_WS="$(cd "$WS_OVERRIDE" && pwd)"
else
  # If script lives in .../rad_workspace/milvus, RAD_WS is parent of milvus/
  RAD_WS="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

MILVUS_DIR="$RAD_WS/milvus"

[[ -d "$RAD_WS" ]]     || { echo "ERROR: RAD_WS not found: $RAD_WS" >&2; exit 1; }
[[ -d "$MILVUS_DIR" ]] || { echo "ERROR: MILVUS_DIR not found: $MILVUS_DIR" >&2; exit 1; }

echo "RAD_WS=$RAD_WS"
echo "MILVUS_DIR=$MILVUS_DIR"
echo "PWD(before)=$(pwd)"
cd "$RAD_WS"
echo "PWD(after)=$(pwd)"

export RAD_WS MILVUS_DIR

# ------------------------------------------------------------
# Choose docker compose command (v2 preferred)
# ------------------------------------------------------------
if docker compose version &>/dev/null; then
  DOCKER_COMPOSE=(docker compose)
elif command -v docker-compose &>/dev/null; then
  DOCKER_COMPOSE=(docker-compose)
else
  echo "docker compose (or docker-compose) not found." >&2
  exit 1
fi

# ------------------------------------------------------------
# Cleanup containers (safe if not present)
# ------------------------------------------------------------
sudo docker rm -f milvus-standalone || true
sudo docker rm -f milvus-etcd || true
sudo docker rm -f milvus-minio || true

# ------------------------------------------------------------
# Optional: wipe persistent volumes for a clean start
# ------------------------------------------------------------
echo "• Clearing persistent data volumes for a clean start..."
sudo rm -rf "$MILVUS_DIR/minio" "$MILVUS_DIR/milvus-data" "$MILVUS_DIR/etcd" || true

echo "• Recreating empty volume directories..."
sudo mkdir -p "$MILVUS_DIR/minio" "$MILVUS_DIR/milvus-data" "$MILVUS_DIR/etcd"

# ------------------------------------------------------------
# Bring up stack
# IMPORTANT: preserve env so ${MILVUS_DIR} substitutions work in compose YAML
# ------------------------------------------------------------
echo "• Starting Milvus stack…"
sudo --preserve-env=RAD_WS,MILVUS_DIR \
  "${DOCKER_COMPOSE[@]}" \
  --env-file "$MILVUS_DIR/compose.env" \
  -f "$MILVUS_DIR/docker-compose.milvus.yml" \
  -f "$MILVUS_DIR/docker-compose.hostnet.yml" \
  up -d

# ------------------------------------------------------------
# Wait for readiness
# ------------------------------------------------------------
echo "• Waiting for Milvus to become ready at http://127.0.0.1:19530 …"
python3 - <<'PY'
import time, sys
from pymilvus import connections, list_collections

uri = "http://127.0.0.1:19530"
for i in range(60):
    try:
        connections.connect(uri=uri)
        list_collections()
        print("Milvus is ready:", uri)
        sys.exit(0)
    except Exception:
        time.sleep(2)

print("Milvus did not become ready in time.", file=sys.stderr)
sys.exit(1)
PY

echo "✓ Milvus is up."
echo "Try:  python3 milvus_connect_test.py --uri http://127.0.0.1:19530"
