SHELL := /bin/bash
.ONESHELL:

REPO_ROOT      := $(shell pwd)
NFS_ROOT       ?= $(shell modelfactory infra get storage.nfsHostRoot 2>/dev/null || echo /data/model-factory-nfs)
# Node name is derived from cluster.yaml when present (used only by optional
# site-repair); empty otherwise. Override with `make NODE_NAME=... <target>`.
NODE_NAME      ?= $(shell modelfactory infra get cluster.nodes.0 2>/dev/null || true)
REGISTRY_HOST  ?= registry.model-factory.svc:5000
# Single-node cluster: build images directly on the node and let kubelet find
# them via containerd's local image store (imagePullPolicy: IfNotPresent).
# The REGISTRY_HOST is reserved for the future deploy-registry target;
# push-images is a no-op until then.
TRAINER_TAG    := nnunet-trainer:0.3.0-ray
TOTALSEG_TAG   := totalseg-trainer:0.3.0-ray
QA_VIEWER_TAG  := model-qa:0.9.2

.PHONY: help
help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

## ─── Bootstrap ──────────────────────────────────────────────────────────────

.PHONY: infra-validate
infra-validate: ## Validate cluster.yaml and print a summary
	modelfactory infra validate

.PHONY: infra-render
infra-render: ## Render k8s manifests from cluster.yaml to .render/infra/ (no apply)
	modelfactory infra render

.PHONY: infra-diff
infra-diff: ## kubectl diff the rendered manifests against the live cluster (read-only)
	modelfactory infra apply --dry-run

.PHONY: bootstrap
bootstrap: ## validate -> discover -> render -> kubectl diff (add --apply to apply). Wrapper: scripts/bootstrap.sh
	@bash scripts/bootstrap.sh

.PHONY: mig-partition
mig-partition: ## Create the MIG layout from cluster.yaml (PRIVILEGED — kills GPU procs)
	modelfactory infra mig-create

.PHONY: repair
repair: ## Run cluster repair (kubelet hostname-override + cloud-init guard) — Brev/GCE only
	@bash infra/cluster-repair/apply.sh

.PHONY: nfs-root
nfs-root: ## Create the NFS-served directory layout
	sudo mkdir -p \
	  $(NFS_ROOT)/datasets \
	  $(NFS_ROOT)/preprocessed \
	  $(NFS_ROOT)/results \
	  $(NFS_ROOT)/weights/totalseg \
	  $(NFS_ROOT)/mlflow-postgres \
	  $(NFS_ROOT)/minio-data \
	  $(NFS_ROOT)/registry-data
	sudo chown -R nvidia:nvidia $(NFS_ROOT)

.PHONY: deploy-infra
deploy-infra: deploy-kueue deploy-mlflow deploy-kuberay deploy-monitoring ## Deploy the full infra stack

.PHONY: mlflow-secrets-check
mlflow-secrets-check: ## Fail fast if infra/kustomize/secrets.yaml has not been created
	@test -f infra/kustomize/secrets.yaml || { \
	  echo "ERROR: infra/kustomize/secrets.yaml missing."; \
	  echo "       cp infra/kustomize/secrets.example.yaml infra/kustomize/secrets.yaml,"; \
	  echo "       fill the admin-password fields, then re-run."; \
	  exit 2; }

.PHONY: deploy-kueue
deploy-kueue: ## Install Kueue + ResourceFlavor + ClusterQueue + LocalQueue
	helm upgrade --install kueue oci://registry.k8s.io/kueue/charts/kueue \
	  --namespace kueue-system --create-namespace \
	  -f infra/helm/kueue/values.yaml --version 0.17.2 --wait
	kubectl create namespace model-factory --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f infra/kustomize/factory-resource-flavor.yaml
	kubectl apply -f infra/kustomize/factory-cluster-queue.yaml
	kubectl apply -f infra/kustomize/factory-local-queue.yaml
	kubectl apply -f infra/kustomize/factory-priority-classes.yaml

.PHONY: deploy-mlflow
deploy-mlflow: mlflow-secrets-check ## Install Postgres + MinIO + MLflow into model-factory namespace
	kubectl create namespace model-factory --dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f infra/kustomize/secrets.yaml
	helm upgrade --install factory-pg bitnami/postgresql \
	  --namespace model-factory -f infra/helm/postgres/values.yaml
	helm upgrade --install factory-minio bitnami/minio \
	  --namespace model-factory -f infra/helm/minio/values.yaml
	helm upgrade --install mlflow community-charts/mlflow \
	  --namespace model-factory -f infra/helm/mlflow/values.yaml

.PHONY: deploy-kuberay
deploy-kuberay: ## Install KubeRay operator + RayCluster with MIG-slice leasing
	# 1) Operator (cluster-scoped CRDs)
	helm repo add kuberay https://ray-project.github.io/kuberay-helm/ 2>/dev/null || true
	helm repo update kuberay
	kubectl create namespace kuberay-system --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install kuberay-operator kuberay/kuberay-operator \
	  --namespace kuberay-system -f infra/helm/kuberay-operator/values.yaml --version 1.6.1 --wait
	# 2) Re-roll Kueue to pick up the ray.io/raycluster integration now that CRDs exist
	helm upgrade --install kueue oci://registry.k8s.io/kueue/charts/kueue \
	  --namespace kueue-system -f infra/helm/kueue/values.yaml --version 0.17.2 --wait
	# 3) MIG-slice configmaps + RBAC + RayCluster
	kubectl apply -f infra/kustomize/factory-mig-uuids-configmap.yaml
	kubectl apply -f infra/kustomize/factory-mig-leases-configmap.yaml
	kubectl apply -f infra/kustomize/factory-claim-mig-script-configmap.yaml
	kubectl apply -f infra/kustomize/factory-ray-rbac.yaml
	kubectl apply -f infra/kustomize/factory-ray-cluster.yaml

.PHONY: deploy-monitoring
deploy-monitoring: ## Install kube-prometheus-stack + Loki + Promtail
	kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -
	helm upgrade --install kps prometheus-community/kube-prometheus-stack \
	  --namespace monitoring -f infra/helm/kube-prometheus-stack/values.yaml
	helm upgrade --install loki grafana/loki-stack \
	  --namespace monitoring -f infra/helm/loki-stack/values.yaml

## ─── Images ─────────────────────────────────────────────────────────────────

.PHONY: build-images
build-images: build-trainer build-totalseg ## Build both trainer images

.PHONY: build-trainer
build-trainer: ## Build the nnunet-trainer image (and import into containerd for kubelet)
	docker build -t $(TRAINER_TAG) -f images/nnunet-trainer/Dockerfile .
	# Kubelet on this node uses containerd, which has a separate image store
	# from docker. Pipe the built image into containerd's k8s.io namespace so
	# `imagePullPolicy: IfNotPresent` Pods can find it without pushing to a
	# registry. ~30s overhead on a re-tag, several minutes on a fresh build.
	docker save $(TRAINER_TAG) | sudo ctr -n=k8s.io images import -

.PHONY: build-totalseg
build-totalseg: build-trainer ## Build the totalseg-trainer (extends trainer)
	docker build -t $(TOTALSEG_TAG) -f images/totalseg-trainer/Dockerfile \
	  --build-arg BASE=$(TRAINER_TAG) .

.PHONY: push-images
push-images: ## Push images to the in-cluster registry (no-op until deploy-registry lands)
	@echo "push-images: no-op — TRAINER_TAG=$(TRAINER_TAG) is a local containerd tag"
	@echo "  (the in-cluster registry isn't deployed yet; kubelet pulls from"
	@echo "   the local containerd store via imagePullPolicy: IfNotPresent)"

## ─── SDK ────────────────────────────────────────────────────────────────────

.PHONY: install-sdk
install-sdk: ## Install the modelfactory CLI/SDK into the current Python env (editable)
	pip install -e ".[dev]"

.PHONY: lint
lint: ## Lint the SDK
	ruff check src/
	mypy src/modelfactory || true

.PHONY: test
test: ## Run unit tests
	pytest tests/ -v

## ─── Verification ──────────────────────────────────────────────────────────

.PHONY: smoke-gpu
smoke-gpu: ## Schedule a 1-GPU pod that prints nvidia-smi -L
	kubectl run gpu-smoke --rm -i --restart=Never --image=nvcr.io/nvidia/cuda:12.5.0-base-ubuntu22.04 \
	  --overrides='{"spec":{"containers":[{"name":"gpu-smoke","image":"nvcr.io/nvidia/cuda:12.5.0-base-ubuntu22.04","command":["nvidia-smi","-L"],"resources":{"limits":{"nvidia.com/gpu":"1"}}}]}}'

.PHONY: smoke
smoke: ## End-to-end smoke: submit a 1-fold MSD-Hippocampus training and verify MLflow + Loki
	bash examples/smoke/run_msd_hippocampus.sh

.PHONY: ray-dashboard
ray-dashboard: ## Port-forward the Ray dashboard to localhost:8265
	kubectl -n model-factory port-forward svc/factory-ray-head-svc 8265:8265

.PHONY: campaign-brain-mr-trio
campaign-brain-mr-trio: ## Fan 045+047+048 × 5 folds out across factory-ray via Ray Tune
	modelfactory campaign run-brain-mr-trio

## ─── QA viewer (unified single-pod app) ────────────────────────────────────

.PHONY: qa-cohort
qa-cohort: ## Materialize the QA validation cohort under /factory/qa-cohort
	modelfactory qa cohort prepare

.PHONY: qa-cohort-preprocess
qa-cohort-preprocess: ## Pre-stage nnUNetv2 preprocessed inputs for every (model, case) pair
	modelfactory qa cohort preprocess

.PHONY: build-qa-viewer
build-qa-viewer: ## Build the unified model-qa image (FastAPI + Next.js static export, NGC PyTorch base)
	docker build -t $(QA_VIEWER_TAG) -f services/qa-viewer/Dockerfile .
	docker save $(QA_VIEWER_TAG) | sudo ctr -n=k8s.io images import -

.PHONY: deploy-qa
deploy-qa: ## Apply the qa-interface kustomize stack (single Deployment, GPU 0)
	kubectl apply -k infra/kustomize/qa-interface

.PHONY: smoke-qa
smoke-qa: ## Smoke-test the QA viewer via its NodePort (32443) directly
	@echo "Hitting the NodePort on localhost:32443 — assumes Service is up."
	@curl -fsS http://localhost:32443/api/healthz | head -c 200; echo
	@echo "  browser:  http://localhost:32443"
	@echo "  public:   set network.qaPublicHost in cluster.yaml + enable ingress (docs/qa.md)"

.PHONY: qa-dev
qa-dev: ## Reminder: run uvicorn + Next.js dev server in two terminals
	@echo "In one terminal:  modelfactory qa server --reload"
	@echo "In another:       cd services/qa-viewer/web && npm install && npm run dev"
	@echo "Then open:        http://localhost:3000  (set NEXT_PUBLIC_QA_API_URL=http://localhost:8080)"

## ─── Teardown (DESTRUCTIVE) ────────────────────────────────────────────────

.PHONY: teardown
teardown: ## Remove the factory stack (NOT the cluster itself)
	-kubectl -n model-factory delete raycluster factory-ray --wait=false
	helm -n monitoring uninstall loki kps || true
	helm -n model-factory uninstall mlflow factory-minio factory-pg || true
	helm -n kueue-system uninstall kueue || true
	helm -n kuberay-system uninstall kuberay-operator || true
	kubectl delete ns model-factory monitoring kueue-system kuberay-system --wait=false || true
