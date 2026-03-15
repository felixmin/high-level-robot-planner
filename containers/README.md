# HLRP Containers

The unified container is now the intended runtime for all stages:

- `containers/Dockerfile.unified`
  Builds one raw-CUDA image with a single Python 3.10 environment at
  `/opt/hlrp-venv` for stage 1, stage 2, and stage 3.
- `containers/requirements.unified.txt`
  Holds the shared Python dependency set, including local installs of
  `lerobot` and `lerobot_policy_hlrp`.

The torch stack is installed explicitly in the Dockerfile so the wheel channel
can stay parameterized by build args. The currently validated unified image uses
`PYTORCH_WHL_CHANNEL=cu128` on both cluster H100 and local RTX 5090 targets.

## Cluster Setup

Cluster presets no longer hardcode a user-specific container path. Set your own
imported unified image in `config/user_config/local.yaml`:

```yaml
cluster:
  container:
    image: /dss/.../enroot/hlrp_unified_cu128_imported_2026-03-14_2248.sqsh
```

The shared cluster configs already set:

```yaml
cluster:
  container:
    python_bin: /opt/hlrp-venv/bin/python
```

so stage 1, stage 2, and stage 3 all use the same interpreter inside the
unified image.

## Refresh Workflow

1. Build on the workstation.
2. Push the image tag to Docker Hub.
3. Import that tag to the cluster with Enroot.
4. Point your `config/user_config/local.yaml` at the imported `.sqsh`.

Example build command:

```bash
docker build -f containers/Dockerfile.unified \
  -t felixmin/hlrp:unified-cuda-cu128 .
```

Example Enroot import target:

```bash
/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/<user>/enroot/hlrp_unified_cu128_imported_<date>.sqsh
```