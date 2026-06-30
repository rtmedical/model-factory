#!/usr/bin/env bash
# Idempotent cluster-repair runner. Re-runnable; safe to invoke on a healthy cluster.
set -euo pipefail

# Node name the kubelet client-cert expects (CN=system:node:<name>).
# Pass as $1 or set MFACTORY_NODE_NAME; this fix is site-specific (Brev/GCE).
NODE_NAME="${1:-${MFACTORY_NODE_NAME:?set MFACTORY_NODE_NAME or pass the node name as arg 1}}"
KUBELET_FLAGS="/var/lib/kubelet/kubeadm-flags.env"
CLOUD_CFG_SRC="$(dirname "$0")/99-preserve-hostname.cfg"
CLOUD_CFG_DST="/etc/cloud/cloud.cfg.d/99-preserve-hostname.cfg"

echo "[1/4] Ensure kubelet has --hostname-override=${NODE_NAME}"
if ! sudo grep -q 'hostname-override=' "${KUBELET_FLAGS}"; then
  sudo cp -a "${KUBELET_FLAGS}" "${KUBELET_FLAGS}.bak-$(date +%Y%m%d-%H%M%S)"
  sudo sed -i "s|\"\$| --hostname-override=${NODE_NAME}\"|" "${KUBELET_FLAGS}"
  echo "    + flag added, restarting kubelet"
  sudo systemctl restart kubelet
else
  echo "    = flag already present"
fi

echo "[2/4] Ensure cloud-init won't rewrite hostname"
if [[ ! -f "${CLOUD_CFG_DST}" ]]; then
  sudo cp "${CLOUD_CFG_SRC}" "${CLOUD_CFG_DST}"
  echo "    + ${CLOUD_CFG_DST} installed"
else
  echo "    = ${CLOUD_CFG_DST} already installed"
fi

echo "[3/4] Wait for node Ready"
for i in {1..30}; do
  status=$(kubectl get node "${NODE_NAME}" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "")
  if [[ "${status}" == "True" ]]; then
    echo "    = node Ready"
    break
  fi
  echo "    . attempt ${i}: status=${status:-unknown}"
  sleep 2
done

echo "[4/4] Label training node"
kubectl label node "${NODE_NAME}" factory.io/training=true --overwrite >/dev/null
echo "    = factory.io/training=true"

echo "Done. GPU allocatable: $(kubectl get node "${NODE_NAME}" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}')"
