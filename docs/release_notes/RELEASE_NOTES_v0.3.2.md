---
sidebar_position: 1
---

# sflow v0.3.2 Release Notes

**Released:** November 2026
**Previous release:** [v0.3.1](https://github.com/NVIDIA/nv-sflow/releases/tag/v0.3.1) (August 2026)

---

## TL;DR

- **sflow is on PyPI as `nv-sflow`.** `pip install nv-sflow` replaces the `git+` install line.
- **Drive a cluster from your laptop.** `--ssh 'user@login'` runs the controller on a login node, with nothing to install there first.
- **Multi-node Docker, no scheduler needed.** List the machines you can SSH into under `hosts:` and sflow uses them like a Slurm partition.

:::warning Before upgrading
Read the [Breaking Changes](#breaking-changes). Most concern installing sflow itself, and one affects **every** backend: unquoted `H:MM:SS` values now load as strings.
Never `pip install sflow` (that name on pypi.org is an unrelated project), and don't `pip uninstall sflow`.
:::

### At a glance

| Area | Change | v0.3.1 | v0.3.2 |
| --- | --- | --- | --- |
| **Install & upgrade** | Install | `uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"` | **`uv pip install nv-sflow`** |
| | Distribution name | `sflow` | **`nv-sflow`** on PyPI (GitHub builds stay `sflow`; import name and CLI unchanged) |
| | `sflow upgrade` default | `main` of the public repo | **latest PyPI release** |
| **Remote execution** | Running on a remote cluster | install sflow there and log in | **`--ssh 'user@login'`** on `run` / `batch` / `compose` |
| **Docker backend** | Multi-node | unavailable | **one `docker run` per host**, supervised as one task |
| | Declaring a host | `docker_host` **and** `ip_address` | **one `address`** |
| | Host missing a peer address | rendezvous hangs with no error | **rejected at plan time**, naming the hosts |
| | Bad host or mount path | found mid-run | **pre-flight** names the host before launch |
| | Containers after a crash | leaked on remote machines | **reclaimed on every host** by the next run |
| | Monitoring a remote pool | sampled the **driver** | collector runs **on each host** |
| **Config & YAML** | `version: "0.1"` | required | **optional** |

---

## Install & upgrade

### Install from PyPI: `nv-sflow`

```bash
uv pip install nv-sflow            # or: pip install nv-sflow
uv pip install "nv-sflow[monitor]" # extras are unchanged
```

- Releases from v0.3.2 on are on pypi.org and the pypi.nvidia.com mirror.
- v0.3.1 and earlier, branches, and pre-release/dev builds stay on git: `sflow upgrade --branch <ref>`.
- **Both names are recognized.** If an environment has both records (for example `pip install nv-sflow` over an old git install), the next `sflow` command keeps the one whose files are on disk and deletes only the other's `*.dist-info`. It touches no package files, uses no network, and prints one line. Set `SFLOW_NO_AUTO_MIGRATE=1` to turn this off.
- Already ran `pip uninstall sflow`? Repair with `pip install --force-reinstall nv-sflow`.

### `sflow upgrade` installs releases

```bash
sflow upgrade                              # latest nv-sflow release
sflow upgrade --sflow-version 0.3.2        # pin a release
sflow upgrade --sflow-version '>=0.3,<0.4' # or a range
sflow upgrade --branch develop             # a git branch or tag instead
sflow upgrade --sflow-index-url <url>      # also search a private index
```

- With uv, only nv-sflow is upgraded (`--upgrade-package`). The rest of a shared environment is left alone.
- A dev build (git checkout, or a `.devN` wheel from a private index) goes back to the PyPI release, even if its version string is higher. Requests are floored at 0.3.2, the first release under the new name.
- When an upgrade crosses the rename, the new files are installed before the stale record is dropped, so sflow is never missing in between.

### `sflow batch` installs the right release

- **Jobs match the submitter.** Without `--sflow-version`, a job installs the sflow version the submitter runs. PyPI/private-index installs now map that version to its git tag (`0.3.1` → `v0.3.1`); before, the job's install failed.
- **Git refs from before the rename still install.** `sflow batch` and `sflow upgrade` install git refs as a bare `git+<url>@<ref>`.

---

## Remote execution: `--ssh`

`run`, `batch` and `compose` can run the sflow controller on an SSH host:

```bash
sflow run workflow.yaml --ssh 'user@login -p 20 -i ~/.ssh/id_ed25519'
sflow batch workflow.yaml --submit --ssh 'user@login -J bastion'
sflow compose base.yaml tasks.yaml -o composed.yaml --ssh 'user@login'
```

- **The remote host only needs Python 3.10+.** sflow uploads its current source (including local edits) and caches a dependency venv on the host. It uploads only the explicit YAML/CSV inputs and the relative `file://` / `fs://` artifacts inside the workspace.
- **Each execution gets one local folder**, `sflow_output/ssh-<timestamp>-<id>/`, with the fetched logs and a `receipt.json`.
- **Reconnect with the receipt.** `sflow watch <receipt>` tails the remote logs and `sflow fetch <receipt> [--all]` copies output back. A dropped connection does not cancel the remote run. Ctrl-C on an attached run does, and `scancel`s any Slurm job it submitted.
- **Options:** `--ssh-follow`, `--ssh-fetch logs|all|none`, `--ssh-remote-root`, `--ssh-tty`.
- **Authentication is left to OpenSSH.** Never put passwords or tokens in `--ssh`. On MFA clusters, open one `ControlMaster` connection and reuse it; sflow prints the recipe when SSH exits with 255.
- The remote run directory is never deleted. Clean up `--ssh-remote-root/runs/` yourself if disk space matters.

Full reference: [CLI → Run through SSH](../user/cli.md#run-through-ssh).

---

## Docker backend

The headline change is **multi-host pools**. Before this release one Docker task could not span machines, so `nodes.count: 2` had nowhere to put the second node. Now you declare the machines once, and sflow treats them like a Slurm partition. The subsections below cover the pool end to end, then the Docker changes that apply to every setup.

### Declare a host pool

Each host needs one **`address`**. sflow derives both the Docker endpoint (`ssh://<address>`, or `ssh://<user>@<address>`) and the address peer containers dial for rendezvous.

```yaml
backends:
  - name: docker_cluster
    type: docker
    default: true
    image: nvcr.io/nvidia/pytorch:24.12-py3
    hosts:
      - name: node-0
        address: 10.0.0.11
        gpus_per_node: 8
      - name: node-1
        address: 10.0.0.12
        user: ubuntu
        gpus_per_node: 8
```

- A task with `resources.nodes.count: 2` is placed on two hosts, as it would be on two Slurm nodes.
- SSH must already work from the driver to each address. Port, key and proxy settings stay in `~/.ssh/config`.
- **`hosts:` is the node list.** A backend-level `nodes:` with a different count is rejected; add or remove hosts instead.
- **`address` can be an SSH alias** (`address: dgx-a`), which picks up user, port, key and `ProxyJump` from `~/.ssh/config`. Containers on other machines can't resolve an alias, so also set `rendezvous_address` to what peers should dial.

Override the derived values only when they really differ:

| Field | Overrides | Use when |
| --- | --- | --- |
| `docker_host` | the derived `ssh://` endpoint | a jump host, or a non-SSH endpoint (`tcp://…`) |
| `context` | the endpoint, via a Docker context | you already manage the daemon as a context |
| `rendezvous_address` | the address peers dial | peers use a separate fabric (e.g. InfiniBand) |

`docker_host` and `context` are mutually exclusive. Each host needs `address`, `docker_host` or `context`. `ip_address` is deprecated (see [Breaking Changes](#breaking-changes)).

**Mixed hardware.** Hosts don't need to be identical. Besides `gpus_per_node`, a host can have its own `mounts` and `extra_args`, **added to** the backend-level ones for that host only:

```yaml
    hosts:
      - name: dgx-a
        address: 10.0.0.11
        mounts:
          - /mnt/fast/models:/models:ro   # this box only
        extra_args: ["--shm-size=32g"]
```

- Each member gets a GPU slice computed for its own host, so members don't share one slice by accident.
- `include_nodes` / `exclude_nodes` filter the pool before placement, to pin an experiment to certain machines or avoid a broken one.

### Pre-flight checks

Before anything launches, sflow checks each host in parallel:

- **The daemon answers:** a bounded `docker version` that exercises SSH auth, the tunnel and the daemon in one call.
- **Bind-mount sources exist on that machine.** Docker silently creates a missing bind source as an empty directory, so a typo would otherwise surface later as "missing data".

How it behaves:

- **One report for the whole pool.** All problems are reported together, each naming its host, so one run shows every fix you need.
- **Shows progress.** Each check is announced before it runs, with its time limit, so a slow pool doesn't look like a hung driver.
- **Never prompts.** Checks run with `BatchMode=yes` and no terminal, so an unknown host key or passphrase-protected key fails immediately instead of opening many parallel SSH prompts.
- Path checks need a shell, so they run only on SSH endpoints (`context:` hosts are skipped).
- Set `SFLOW_SKIP_DOCKER_PREFLIGHT=1` to skip the checks that touch hosts.

### How a multi-host task runs

sflow sends one `docker run` per host and supervises them from the driver as a single task.

- **Each member knows its place:** `SFLOW_TASK_NODE_INDEX`, `SFLOW_TASK_NUM_NODES` and `SFLOW_LEADER_ADDRESS`. Rank 0 is always the first assigned host.
- **One merged task log**, each line labelled with node name and rank. Probes read a single stream and don't need to know the node count.
- **Fail fast.** The first member to exit non-zero fails the task with its exit code, and the other members are stopped right away.
- **No silent hangs.** A task whose hosts can't all supply a rendezvous address is rejected at plan time, naming them.
- Environment values are passed with `-e KEY`, so secrets don't appear on the `docker run` command line.
- **`--dry-run --verbose` shows the whole fan-out**, one `docker run` per member with its rank and target daemon:

```
fan-out: 2 commands, one per member
   • node node-0 (rank 0) -> ssh://10.0.0.11
     docker --host ssh://10.0.0.11 run --rm --name sflow-...-node-0 ...
   • node node-1 (rank 1) -> ssh://10.0.0.12
     docker --host ssh://10.0.0.12 run --rm --name sflow-...-node-1 ...
```

### Cleanup

- **Every member container is removed** on completion, failure, signal or Ctrl-C.
- **After a driver crash, the next run cleans up leftovers on every host**, not just the local daemon. Container names now include the driver's machine identity as well as its PID, so the sweep never touches a colleague's containers on a shared daemon (this renames containers, see [Breaking Changes](#breaking-changes)).
- Teardown no longer blocks other tasks' probes and log streaming while it waits on remote SSH calls.
- **Running the driver inside a container (e.g. CI)?** Set `SFLOW_DRIVER_ID` to a stable value. Otherwise each run gets a new machine identity and can't reclaim its own previous leftovers.

### Hardware monitoring on a pool

`monitor:` now works with multi-host `hosts:` pools. The collector runs **on each host** over SSH (bare metal, no container) and streams samples back to the driver. Before, it ran on the driver and reported the driver's hardware under the hosts' names.

- **Nothing is copied to or left on the host.** The collector source is sent over stdin.
- **The host needs `python3`** (standard library only), plus `nvidia-smi` for the `gpu` scope. The workload image doesn't matter. Images whose entrypoint isn't a shell (e.g. `vllm/vllm-openai`) no longer break monitoring.
- **Missing `python3`?** The collector exits with a message, and the run still succeeds with no samples for that node.
- **Hosts without an SSH endpoint** (`context:` or a `tcp://` `docker_host:`) run the collector as a container from the backend's `image:`, which then needs `python3`. If the pool declares no GPUs, the `gpu` scope is skipped and the other four scopes are collected.
- **A collector that fails to start now logs a warning.** Before, the run finished "successfully" with an empty report and nothing above INFO.
- **Collectors stop in parallel at teardown**, so stopping them no longer adds one network round trip per host after the workflow is done.

### Other Docker changes

These apply to single-host setups too.

- **Local `fs://` / `file://` artifacts are auto-mounted** at the same path, as srun always did. Recipes that worked on Slurm no longer fail on Docker with a missing file.
- **`image` is optional on the backend.** It's only used by tasks without an `operator:` and by the container-based monitor collector. If either needs it and it's missing, the error names the field.
- **`docker_run` gains `rm: false`** to keep a container after it exits for `docker logs` / `docker inspect`. Teardown stops it instead of removing it, but the next `sflow run` on that host still cleans it up, so inspect it first.
- **No more false dry-run mount warnings for remote hosts.** Their mounts resolve on other machines, so checking the controller was meaningless. The artifact-path warning now says paths must exist *on the execution hosts*. (Kubernetes gets the same fix.)

### New Docker samples

| Sample | What it shows |
| --- | --- |
| `self_contained/docker/multi_node` | two-host `torchrun` smoke test; prints two ranks only if node index and leader address are wired correctly |
| `self_contained/docker/dynamo_vllm_agg` | aggregated serving on a host pool |
| `self_contained/docker/dynamo_vllm_disagg` | cross-node prefill/decode disaggregation on a host pool |
| `self_contained/docker/sglang_glm52` | GLM-5.2 NVFP4, TP8 on one remote host, with the InferenceX AgentX benchmark |
| `self_contained/docker/sglang_disagg_glm53` | GLM-5.3-Flash prefill/decode disaggregation across a pool (SGLang) |
| `self_contained/docker/vllm_disagg_glm53` | same, with vLLM |

---

## Config & YAML

- **Slurm walltimes no longer need quotes.** PyYAML follows YAML 1.1, which read `10:00:00` as the base-60 integer `36000`, and Slurm took that as minutes. sflow now uses YAML 1.2 scalar rules, so `time: 10:00:00` equals `time: "10:00:00"`, and `10:00:00.5` no longer fails validation. A bare integer (`time: 5400`) still means minutes, as in `sbatch --time`. This applies to every field, not just `time` (see [Breaking Changes](#breaking-changes)).
- **`version` is optional.** Leaving it out means `"0.1"`. If you set it, it must still be `"0.1"`, and merged files must agree.
- **`sflow batch` directory scans find recipes by their `workflow:` section**, not by `version:`. Recipes without `version` are found, and fragments such as a backends-only file are no longer submitted on their own.

---

## Other improvements

### Logs

- **Offloaded task logs have no ANSI escape codes.** The `bash`/`srun`/`docker_run` log prefixer strips colour and cursor codes as it writes, and Kubernetes merged-pod logs are stripped live. Readiness patterns like `endpoint=generate` now match logs that tools such as dynamo colour even without a TTY.

### Slurm and modular samples

- **`modular/backend_agnostic` gains a Slurm MPI fragment.** `backends/slurm_mpi.yaml` runs one MPI rank per GPU under pmix (for TRT-LLM); `backends/slurm.yaml` stays single-process for vLLM/SGLang. This mirrors `k8s.yaml` vs `k8s_mpi.yaml`.
  - The model is now an `fs://` artifact on the backend fragment and is mounted automatically. Override it with `-a LOCAL_MODEL_PATH=fs:///path`, not `-s MODEL_PATH=...`.
  - The TRT-LLM workload exposes `MAX_BATCH_SIZE` and `KV_FREE_GPU_MEMORY_FRACTION`.
  - Kubernetes samples drop the placeholder `node_selector: {tenant: your-node-tenant}` that had to be deleted before they would schedule.
- **Slurm dynamo vLLM samples run on current images and GB300.** `self_contained/slurm/dynamo_vllm_agg` and `dynamo_vllm_disagg` move from `vllm-runtime:0.8.0` to `vllm-runtime:1.2.1-cuda13` (0.8.0's Triton can't target `sm_103a`). The disagg sample uses `--disaggregation-mode prefill|decode` with NixlConnector `--kv-transfer-config` instead of `--is-prefill-worker`. Override `DYNAMO_IMAGE` to pin another image.

### Agent skills

- `check_gpu_plan.py` evaluates `${{ }}` arithmetic with a restricted AST evaluator instead of `eval()`, so a recipe can't run code through it.
- `preflight.py` and `validate_sflow_yaml.py` document their exit codes (`2` = usage error).
- `parse_sflow_errors.py` explains Jinja2 syntax errors more clearly.
- Skill docs use the correct `${{ }}` syntax throughout.

---

## Breaking Changes

Grouped by area; within each, the most likely to affect a v0.3.1 setup comes first.

### Install & upgrade

1. **The distribution is renamed `sflow` → `nv-sflow`.**
   Tools that name the distribution must use the new name: `pip show`, `pip uninstall`, requirement files, `importlib.metadata.version("sflow")`. GitHub builds (`git+https://github.com/...`) still report `sflow`.
   **Fix:** `pip install nv-sflow`, or `sflow upgrade` (twice from v0.3.1 or earlier). The leftover `sflow` record is removed on the next `sflow` run. Do **not** `pip uninstall sflow` first, because both names own the same files.

2. **Wheels up to v0.3.1 are named `sflow`**, so `--sflow-index-url <url> --sflow-version 0.3.1` now looks for `nv-sflow==0.3.1` and fails.
   **Fix:** install old releases from their git tag: `sflow batch --sflow-version v0.3.1` (no index URL) or `sflow upgrade --branch v0.3.1`.

3. **`sflow upgrade` defaults to the latest PyPI release**, not `main`.
   **Fix:** `sflow upgrade --branch main` for the old behavior.

4. **`sflow upgrade --sflow-version` accepts only PyPI versions.** `develop` or `<repo>@<ref>` is rejected with an error that names the replacement.
   **Fix:** `--sflow-version <ref>` → `--branch <ref>`; `--sflow-version <repo>@<ref>` → `--repo <repo> --branch <ref>`. (`sflow batch --sflow-version` is unchanged and still takes a git ref.)

### Config & YAML (all backends)

5. **Unquoted `H:MM:SS` values are strings everywhere in a config**, not just in `time`. `DURATION: 1:30` used to load as `90`; it now loads as `"1:30"`.
   **Fix:** write the number you mean (`DURATION: 90`).

### Docker backend

6. **`nodes: N>1` requires a `hosts:` pool.** Before, sflow invented fake `localhost-N` nodes on one machine and gave the same GPUs out twice.
   **Fix:** list the machines under `hosts:`, or use `nodes: 1` to run several containers on this machine.

7. **Container names changed:** `sflow-p<pid>-<task>-<node>` → `sflow-<machine>p<pid>-<task>-<node>`, where `<machine>` is a short hash of the driver's machine identity. This keeps drivers on different machines from cleaning up each other's containers on a shared daemon.
   **Fix:** update scripts that match the old name. Containers left by sflow before the upgrade don't match the new pattern; remove them once by hand.

8. **`ip_address` on a host is deprecated** in favour of `rendezvous_address`, or just `address` if the machine has one address. It still works and logs a warning. If `address` and `ip_address` differ, `address` is where the driver sends `docker run` and `ip_address` is what peers dial.
   **Fix:** drop `ip_address`, or rename it to `rendezvous_address` if peers really dial a different address.

### Kubernetes backend

9. **Dry-runs no longer warn about container mount paths.** Those paths live inside the cluster, so the check was never meaningful.
   **Fix:** none needed.

### Release process (maintainers only)

10. **Release tags must use PEP 440 normal form.** `v0.4.0-rc1` is rejected.
    **Fix:** tag `v0.4.0rc1`.

---

## Tested Environments & Current Support

**Validated on**

- vanilla bare-metal Kubernetes
- Google Kubernetes Engine (GKE)
- Slurm (GRES and non-GRES partitions)
- a static pool of SSH-reachable Docker hosts (new)

**Known limitations**

*Docker host pools*

- **sflow is not a distributed-compute scheduler.** It starts containers and supplies node metadata. Process ranks, NCCL/UCX transport and the launcher stay the workload's job. Nothing queues, backfills or preempts; the pool is declared up front.
- **No cross-host GPU lease.** GPU reservations are tracked per machine, so tasks on a `hosts:` pool use the planner's device slice and log a warning. Own the hosts exclusively, or coordinate outside sflow.
- **Controller paths are not auto-mounted on remote hosts.** Declare them under `mounts:` (backend or per host).

*Kubernetes*

- **No hardware monitoring.** There is no DCGM/DaemonSet collector, so `monitor:` blocks are skipped. Unchanged since v0.3.0.

*All backends*

- **`--dry-run` prints full commands only for a Docker fan-out.** Other backends show the plan (allocation, operator config, placement) without per-task commands.

---

## Documentation

**User docs**

- **CLI → Run through SSH:** full `--ssh` reference.
- **[Backends → Docker](https://nvidia.github.io/nv-sflow/docs/user/backends):** `hosts:` inventory, host identity and overrides, pre-flight, reusing `~/.ssh/config`.
- **[Monitor](https://nvidia.github.io/nv-sflow/docs/user/monitor):** monitoring a multi-host pool and what each host needs.
- **Backends / Configuration / Quick Reference:** unquoted walltimes and optional `version`.
- **Outputs:** the Kubernetes log section now matches v0.3.1's behavior (streamed log, no post-run re-fetch).

**Docs site**

- **[vLLM Recipes command builder](https://nvidia.github.io/nv-sflow/vllm-recipes):** a page that builds the `sflow` command for a vLLM recipe, including Docker bundles that work under any launcher and follow the live nightly image tag. Fixes since the first version:
  - Downloaded self-contained YAML keeps the namespace, PVC, local-model, cache, interface and Slurm settings from the page.
  - Copied commands and bundle scripts quote paths and values with spaces or shell metacharacters.
  - Recipes without cache redirection disable that option; read-only PVC cache destinations show the required volume change.
  - Failed asset requests show a Retry button.
  - The first load fetches a gzip index (~16 kB instead of 286 kB).
  - Docker/Kubernetes bundles for 39 single-node recipes no longer fail with `unknown flag: --export`.
  - DeepSeek-V4.1-Flash recipes use `vllm/vllm-openai:nightly`, replacing a pinned nightly tag removed from Docker Hub.
- **Project page** refreshed, with a Chinese version.

**Developer docs**

- `docs/developer/dev-notes/docker-multi-host.md`: how one task spans several Docker hosts, why host identity is a single `address`, and what sflow deliberately doesn't do.

Full documentation: [nvidia.github.io/nv-sflow](https://nvidia.github.io/nv-sflow/)

---

## Upgrade Guide

```bash
pip install nv-sflow          # or: sflow upgrade
                              # (from v0.3.1 or earlier, run it twice:
                              #  GitHub build first, then PyPI)

sflow --version               # first run tidies up the old 'sflow' record

# Re-plan every recipe without using resources. This catches Docker
# nodes > 1 without hosts:, and variables that relied on H:MM as a number.
sflow run -f your_recipe.yaml --dry-run

# New samples worth a look
sflow sample self_contained/docker/multi_node
sflow sample self_contained/docker/dynamo_vllm_disagg
```

Checklist:

- [ ] Replace `sflow @ git+...` in scripts, Dockerfiles and requirement files with `nv-sflow` (or `nv-sflow==0.3.2` to pin).
- [ ] Replace `sflow upgrade --sflow-version <ref>` with `sflow upgrade --branch <ref>`.
- [ ] Install pinned old releases from their git tag, not a private index (Breaking Change 2).
- [ ] `--dry-run` every recipe and fix `H:MM` variables (Breaking Change 5) and Docker `nodes: > 1` without `hosts:` (Breaking Change 6).
- [ ] Update tooling that matches `sflow-p<pid>-` container names, and remove old containers once by hand (Breaking Change 7).
- [ ] Moving to a host pool? Give each machine one `address`, drop `docker_host` / `ip_address` unless they really differ, and let pre-flight report what's unreachable.

Repository: [https://github.com/NVIDIA/nv-sflow](https://github.com/NVIDIA/nv-sflow)
