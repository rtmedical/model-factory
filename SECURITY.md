# Security Policy

## Reporting a vulnerability

Please report security issues privately via GitHub's "Report a vulnerability"
(Security → Advisories) on this repository, rather than opening a public issue.
We aim to acknowledge reports within a few business days.

## Scope and operational notes

model-factory deploys cluster infrastructure and handles credentials. A few
things to keep in mind when running it:

- **Secrets** (MLflow, MinIO/S3, Postgres) are Kubernetes Secrets. Never commit
  `secrets.yaml` or `cluster.yaml` — both are git-ignored. Use the
  `secrets.example.yaml` template.
- **MinIO / MLflow / Grafana** ship with admin credentials you must change at
  deploy time. Do not expose them publicly without an authenticating proxy.
- **The QA viewer** can be exposed via NodePort or Ingress. Put it behind
  authentication before exposing it on a public hostname.
- **Patient data**: this tool processes medical images. You are responsible for
  ensuring your deployment meets the privacy/compliance obligations
  (de-identification, access control, audit) that apply to your data.
- **MIG partitioning** (`modelfactory infra mig-create`) runs privileged
  `nvidia-smi` and will terminate running GPU processes — run it intentionally.
