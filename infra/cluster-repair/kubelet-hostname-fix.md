# Cluster repair: kubelet hostname binding

## Problem (observed 2026-05-12)

`kubectl get nodes` showed the single node `<node-name>` as
**NotReady**. The kubelet was still active (`systemctl is-active kubelet` →
active) but had stopped posting status to the API server six days earlier.

Kubelet log showed RBAC denials for every operation:

```
nodes "<os-hostname>" is forbidden:
  User "system:node:<node-name>" cannot get resource "nodes"
  ...node '<node-name>' cannot read '<os-hostname>',
     only its own Node object
```

## Root cause

The Brev cloud-init `91-gce-system.cfg` datasource renamed the OS hostname from
`<node-name>` to `<os-hostname>` on 2026-05-06 17:49 UTC.
Kubelet's client certificate is CN-locked to the original name:

```
$ openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -subject
subject=O = system:nodes, CN = system:node:<node-name>
```

The Node-Authorizer admission plugin enforces that a kubelet identified by that
cert may only mutate the Node object of the same name — so the rename created
an inescapable catch-22: kubelet can't update the old Node (it advertises a
different name) and can't create a new Node or CSR under the new name (Node
Authorizer / NodeRestriction reject it because CN doesn't match).

## Fix

Two layers:

### 1. `--hostname-override` on kubelet (applied)

`/var/lib/kubelet/kubeadm-flags.env`:

```bash
KUBELET_KUBEADM_ARGS="--container-runtime-endpoint=unix:///run/containerd/containerd.sock \
                      --pod-infra-container-image=registry.k8s.io/pause:3.10 \
                      --hostname-override=<node-name>"
```

```bash
sudo systemctl restart kubelet
```

Node went **Ready** within ~30 seconds and all 8 GPUs became allocatable.

### 2. Prevent future hostname drift (`99-preserve-hostname.cfg`)

Copy `99-preserve-hostname.cfg` (sibling file in this directory) to
`/etc/cloud/cloud.cfg.d/99-preserve-hostname.cfg`. This stops cloud-init from
rewriting `/etc/hostname` on subsequent boots so the OS hostname will not drift
again.

```bash
sudo cp 99-preserve-hostname.cfg /etc/cloud/cloud.cfg.d/
```

## Verification

```bash
kubectl get nodes                                            # → Ready
kubectl describe node | grep nvidia.com/gpu                  # → 8 allocatable
sudo journalctl -u kubelet -n 50 --no-pager | grep -i error  # → no RBAC denials
```

## Recovery if it happens again

The fix is in `/var/lib/kubelet/kubeadm-flags.env` on the persistent root
filesystem. If a future Brev rehydration wipes that file, re-apply the
`--hostname-override` flag and `systemctl restart kubelet`. The kubelet client
cert is valid until **2027-04-27** (`openssl x509 -in /var/lib/kubelet/pki/kubelet-client-current.pem -noout -dates`),
after which `kubeadm certs renew` will rotate it.
