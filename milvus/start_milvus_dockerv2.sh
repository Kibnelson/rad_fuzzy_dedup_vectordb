set -euo pipefail

COMPOSE_FILE="docker-compose.milvus.yml:docker-compose.hostnet.yml"

sudo docker rm -f milvus-standalone || true
sudo docker rm -f milvus-etcd || true
sudo docker rm -f milvus-minio || true



# --- NEW SECTION ---
echo "• Clearing persistent data volumes for a clean start..."
sudo rm -rf /home/nelson/rad_workspace/milvus/minio || true
sudo rm -rf /home/nelson/rad_workspace/milvus/milvus-data || true

echo "• Recreating empty volume directories..."
sudo mkdir -p /home/nelson/rad_workspace/milvus/minio
sudo mkdir -p /home/nelson/rad_workspace/milvus/milvus-data
# --- END NEW SECTION ---


if docker compose version &>/dev/null; then
  DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &>/dev/null; then
  DOCKER_COMPOSE="docker-compose"
else
  echo "docker compose (or docker-compose) not found." >&2
  exit 1
fi

echo "• Starting Milvus stack…"
sudo ${DOCKER_COMPOSE} \
  --env-file /home/nelson/rad_workspace/milvus/compose.env \
  -f /home/nelson/rad_workspace/milvus/docker-compose.milvus.yml -f /home/nelson/rad_workspace/milvus/docker-compose.hostnet.yml up -d

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

