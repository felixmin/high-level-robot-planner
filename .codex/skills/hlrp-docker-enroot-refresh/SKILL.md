---
name: hlrp-docker-enroot-refresh
description: "Build and refresh the HLRP LeRobot container workflow end-to-end: prune local Docker state, build and push `felixmin/hlrp:latest` from `containers/Dockerfile.lerobot` using the bundled `lerobot/` repo in this workspace as build context, prune Docker again, import the image to the LRZ cluster as an Enroot `.sqsh`, and safely swap it into `/dss/dssmcmlfs01/pn57pi/pn57pi-dss-0001/felix_minzenmay/enroot/lam.sqsh` by renaming the old file instead of deleting it. Use when the user asks to rebuild, publish, import, replace, or debug the HLRP stage-3 container."
---

# HLRP Docker Enroot Refresh

Use this workflow when refreshing the cluster container from the workstation. Keep the local Docker steps on the workstation, keep cluster work behind `ssh ai`, never delete cluster images, and prefer renaming backups in place.

Read [workflow.md](references/workflow.md) for the defaults, paths, and failure modes. Use the scripts in `scripts/` rather than rewriting the commands.

## Workflow

1. Verify execution context with `pwd` and `hostname`.
2. Run `scripts/build_push_prune.sh` from the workstation.
3. Submit the Enroot import with `scripts/submit_enroot_import.sh`.
4. Monitor the returned Slurm job with `squeue` and `sacct`.
5. If the import OOMs, keep the partial file, choose a new output path, and rerun with higher memory.
6. Before swapping images, check `squeue --me` and avoid replacing `lam.sqsh` while jobs are active unless the user explicitly wants that.
7. Swap the imported image into place with `scripts/swap_enroot_image.sh`. This renames the existing `lam.sqsh` to a timestamped backup and moves the new file into `lam.sqsh`.

## Rules

- Use `containers/Dockerfile.lerobot` and the bundled `lerobot/` repo in this workspace as the Docker build context.
- Use the Enroot URI `docker://felixmin/hlrp:latest`. Do not use `docker://docker.io/felixmin/hlrp:latest`.
- Treat OOM during `enroot import` as a memory tuning issue, not a registry issue.
- Default Enroot import settings should be `--mem=128G`, `-c 4`, and `ENROOT_MAX_PROCESSORS=4`.
- Never delete cluster images or partial outputs. Rename or leave them in place.
- The cluster configs point at `.../enroot/lam.sqsh`, so replacing the active image means swapping a new file into that exact path.

## Resources

- `scripts/build_push_prune.sh`
  Run the local Docker prune -> build -> push -> prune workflow. Defaults match the bundled LeRobot build path.
- `scripts/submit_enroot_import.sh`
  Submit the high-memory `enroot import` job on `ssh ai` and print the Slurm job id.
- `scripts/swap_enroot_image.sh`
  Rename the old `lam.sqsh` to a timestamped backup and move the new image into place.
- `references/workflow.md`
  Read this for the validated command shapes, default paths, and known failure modes.

## Validation

- Run the scripts with `--dry-run` first when adapting paths or tags.
- After editing the skill, run `quick_validate.py` on the skill folder.
