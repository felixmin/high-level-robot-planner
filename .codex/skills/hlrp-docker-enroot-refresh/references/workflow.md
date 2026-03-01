# HLRP Docker Enroot Refresh Workflow

## Defaults

- Local repo root: `/mnt/data/workspace/code/high-level-robot-planner`
- Dockerfile: `/mnt/data/workspace/code/high-level-robot-planner/containers/Dockerfile.lerobot`
- Docker build context: `/mnt/data/workspace/code/high-level-robot-planner`
- Pushed tag: `felixmin/hlrp:latest`
- Cluster host: `ai`
- Cluster Enroot dir: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot`
- Active cluster image path: `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/lam.sqsh`

## Validated Command Shapes

Local build/push:

```bash
docker system prune -a --volumes -f
docker builder prune -a -f
docker build \
  -f /mnt/data/workspace/code/high-level-robot-planner/containers/Dockerfile.lerobot \
  -t felixmin/hlrp:latest \
  /mnt/data/workspace/code/high-level-robot-planner
docker push felixmin/hlrp:latest
docker system prune -a --volumes -f
docker builder prune -a -f
```

Cluster import:

```bash
sbatch -p lrz-cpu -q cpu -t 01:00:00 --mem=128G -c 4 -J enroot-import-hlrp-oli \
  --wrap "mkdir -p /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot && \
  export ENROOT_MAX_PROCESSORS=4 && \
  enroot import -o /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/<new-name>.sqsh \
  docker://felixmin/hlrp:latest && \
  ls -lh /dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/<new-name>.sqsh"
```

Safe swap:

```bash
mv lam.sqsh lam_YYYY-MM-DD_pre_refresh.sqsh
mv <new-name>.sqsh lam.sqsh
```

## Known Failure Modes

- `Dockerfile.lerobot` expects the bundled `lerobot/` folder to be present in the repo root build context.
- `docker://docker.io/felixmin/hlrp:latest` is the wrong Enroot URI. Use `docker://felixmin/hlrp:latest`.
- `enroot import` can OOM while creating squashfs. The validated fix was `--mem=128G`, `-c 4`, and `ENROOT_MAX_PROCESSORS=4`.
- If an import job OOMs, the partial `.sqsh` may exist but should be treated as invalid. Do not delete it; create a new output filename for the retry.
- Do not replace `lam.sqsh` while jobs are active unless the user explicitly accepts that risk.
