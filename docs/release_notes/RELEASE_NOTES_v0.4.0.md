# sflow v0.4.0 Release Notes

**Release date:** TBD
**Previous release:** [v0.3.2](https://github.com/NVIDIA/nv-sflow/releases/tag/v0.3.2)

---

## New Features

### 1. Pull each container image once per node, before any task starts — `prefetch_images`

Every task used to pull its own container image when it started. Tasks that shared an image pulled it several times, and a failed pull failed one task mid-run, often after the expensive tasks were already up. On Slurm, for example, enroot imports of a large NIM image on fresh nodes intermittently hit registry 401s. Each extra import was another chance to fail, and the failure took the whole workflow down late.

Set `prefetch_images: true` on any backend (off by default) and sflow pulls each task image **once per node that will run it**, after allocation and before the first task launches:

```yaml
backends:
  - name: gpu_cluster
    type: slurm        # or docker / kubernetes
    prefetch_images: true
```

- **Retries:** a failed pull is retried twice, 10 s apart, on the failed nodes only.
- **Fails before any task starts:** if the last attempt fails, sflow names the image, the node(s) and the runtime's output, releases the allocation and exits.
- **Per backend:**
  - Slurm runs one `srun --overlap --container-image=<image> true` step per image.
  - Docker runs `docker image inspect` and, only if the image is missing, `docker pull` on each host's daemon.
  - Kubernetes runs one short pod per node, pinned by hostname. The pod holds no GPU and uses the tasks' pull secrets and policy.
- **Skipped:** local `.sqsh` and path images, tasks that run no container, and the `local` backend.
- **Dry run:** `--dry-run` pulls nothing and lists the planned `image -> nodes` under **Image prefetch**.

The `dynamo_vllm_agg` samples for Slurm, Docker and Kubernetes enable it. See [Image prefetch](../user/backends.md#image-prefetch-all-backends).
