---
sidebar_position: 1
---

# sflow v0.3.2 Release Notes

**Release date:** TBD
**Previous release:** [v0.3.1](https://github.com/NVIDIA/nv-sflow/releases/tag/v0.3.1) (August 2026)

---

## Highlights

v0.3.2 is about where sflow comes from and where it runs. **sflow is on PyPI**: the distribution is now named **`nv-sflow`**, so `pip install nv-sflow` replaces the `git+` install line. **sflow can drive a cluster from your laptop**: `--ssh` moves the controller to a login node without installing anything there first. **sflow can drive a multi-node run without a scheduler**: if you have a handful of lab machines you can already SSH into, that is now a cluster sflow can drive. And a long-standing YAML trap is closed: an unquoted `time: 10:00:00` no longer silently becomes a 25-day Slurm request.

The Docker backend could place a *single* container on a chosen remote host, but one task could not span several. A task asking for `nodes.count: 2` had nowhere to put the second node, so on Docker multi-node was simply unavailable — the options were a hand-rolled SSH script or installing Slurm for four boxes. Declaring the hosts was its own puzzle: each wanted both a `docker_host` and an `ip_address`, nothing explained that they answer different questions, and a host missing `ip_address` silently produced no leader address, so the workload hung in rendezvous with no error at all.

Now you declare the machines once — **one `address` per host** — and sflow treats that inventory the way it treats a Slurm partition. It sends one `docker run` per assigned host over the SSH connection you already have, and supervises them as a single task: merged rank-labelled logs, fail-fast with peer cancellation, and no container left behind on any machine. **Hardware monitoring follows the pool**, running the collector directly on each host over the same SSH connection instead of sampling the driver and reporting its hardware under your nodes' names.

| Area | v0.3.1 | v0.3.2 |
| --- | --- | --- |
| **Install** | `uv pip install "sflow @ git+https://github.com/NVIDIA/nv-sflow.git@main"` | **`uv pip install nv-sflow`** |
| **Distribution name** | `sflow` | **`nv-sflow`** on PyPI (GitHub builds stay `sflow`; import name and CLI unchanged) |
| **Running on a remote cluster** | install sflow there and log in | **`--ssh 'user@login'`** on `run` / `batch` / `compose` |
| **Docker multi-node** | unavailable — one task could not span hosts | **one `docker run` per assigned host**, supervised as one task |
| **Declaring a Docker host** | `docker_host` **and** `ip_address`, both required, neither explained | **one `address`**; endpoint and rendezvous both derived from it |
| **A host missing its peer address** | no leader address, silent rendezvous hang | **rejected at planning time**, naming the hosts |
| **A bad host or mount path** | discovered mid-run, error named a container | **pre-flight** names the host before anything launches |
| **A member failing** | n/a | first non-zero exit **fails the task with its code**; peers stopped |
| **Containers after a crash** | leaked on remote machines | **reclaimed on every host** by the next run |
| **Heterogeneous pools** | n/a | per-host `gpus_per_node`, `mounts`, `extra_args` |
| **Monitoring a remote pool** | collector sampled the **driver** | collector runs **on each host**, streams samples back |
| **Unquoted `time: 10:00:00`** | read as 36000 → **36000 minutes** | **10 hours**, same as the quoted form |
| **`version: "0.1"`** | required in every file | **optional** |
| **`sflow upgrade` default** | `main` of the public repo | **latest PyPI release** |

> **Upgrading:** `pip install nv-sflow`, or `sflow upgrade`. From v0.3.1 or earlier, run `sflow upgrade` twice: the old command lands on the current GitHub build, which then moves you to PyPI. Never `pip install sflow` — that name on pypi.org is an unrelated project — and don't `pip uninstall sflow`. **Read the [Breaking Changes](#breaking-changes) before upgrading.** Several concern installing sflow itself; one Docker config that used to be accepted is now rejected at plan time (`nodes: N>1` with no `hosts:`), and one YAML-loading change (unquoted `H:MM:SS` values) affects **every** backend, Slurm included — not just Docker.

---

## New Features

### 1. Install from PyPI — `nv-sflow`

```bash
uv pip install nv-sflow            # or: pip install nv-sflow
uv pip install "nv-sflow[monitor]" # extras are unchanged
```

Releases from v0.3.2 on are published to pypi.org (and the pypi.nvidia.com mirror); v0.3.1 and earlier stay on git (`sflow upgrade --branch v0.3.1`), and pre-release or dev builds never go to PyPI. Branches stay installable from git (`sflow upgrade --branch <ref>`).

**One install, whichever name it came under.** The package is `nv-sflow` on PyPI and `sflow` in GitHub builds and older releases; sflow recognises both. When an environment ends up holding both records — `pip install nv-sflow` over an old git install — the next `sflow` command keeps the one whose files are on disk and deletes only the other's `*.dist-info`: no package files touched, no network, one line of output. `SFLOW_NO_AUTO_MIGRATE=1` turns it off. If `pip uninstall sflow` ran before that, `pip install --force-reinstall nv-sflow` repairs it.

### 2. Run through SSH — `--ssh`

`run`, `batch` and `compose` can move the sflow controller to an SSH host:

```bash
sflow run workflow.yaml --ssh 'user@login -p 20 -i ~/.ssh/id_ed25519'
sflow batch workflow.yaml --submit --ssh 'user@login -J bastion'
sflow compose base.yaml tasks.yaml -o composed.yaml --ssh 'user@login'
```

- **Nothing to install remotely** beyond Python 3.10+. sflow uploads its own current source (including local edits), caches a dependency venv on the host, and uploads only the explicit YAML/CSV inputs plus relative `file://` / `fs://` artifacts inside the workspace.
- **One local folder per execution**, `sflow_output/ssh-<timestamp>-<id>/`, holding the fetched logs and a `receipt.json`.
- **Reconnect after a dropped connection** with the receipt: `sflow watch <receipt>` tails the remote logs, `sflow fetch <receipt> [--all]` copies output back. A dropped connection does not cancel the remote run; Ctrl-C on an attached run does (and `scancel`s any Slurm job it submitted).
- **Tunable:** `--ssh-follow`, `--ssh-fetch logs|all|none`, `--ssh-remote-root`, `--ssh-tty`.
- **Authentication stays OpenSSH's job.** Never put passwords or tokens in `--ssh`. On MFA clusters, open one `ControlMaster` connection and reuse it; sflow prints that recipe when SSH exits with 255.

The remote run directory is never deleted, so clean up old sessions under `--ssh-remote-root/runs/` if disk space matters. Full reference: [CLI → Run through SSH](../user/cli.md#run-through-ssh).

### 3. `sflow upgrade` installs releases

```bash
sflow upgrade                              # latest nv-sflow release
sflow upgrade --sflow-version 0.3.2        # pin a release
sflow upgrade --sflow-version '>=0.3,<0.4' # or a range
sflow upgrade --branch develop             # a git branch or tag instead
sflow upgrade --sflow-index-url <url>      # also search a private index
```

With uv it upgrades only nv-sflow (`--upgrade-package`) and leaves the rest of a shared environment alone. A dev build (a git checkout or a `.devN` wheel from a private index) goes back to the PyPI release even when its version string is higher; the request is floored at 0.3.2, the first release under the new name. When an upgrade crosses the rename, the new files are installed first and the stale record is dropped afterwards, so sflow is never missing in between.

### 4. Multi-host Docker — declare the machines, run across them

A static inventory of Docker hosts. Each is identified by **`address`**: the machine. sflow derives the Docker control endpoint (`ssh://<address>`, or `ssh://<user>@<address>`) *and* the address peer containers dial to rendezvous, so the common case is one line per machine.

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

A task asking for `resources.nodes.count: 2` is placed on two hosts from that pool, exactly as it would be on two Slurm nodes. SSH must already work from the driver to each address; port, key and proxy settings stay in `~/.ssh/config` rather than being mirrored into the recipe.

**`hosts:` is the node list.** A backend-level `nodes:` asking for a different count is rejected rather than quietly ignored — add or remove hosts to change the node count.

**`address` may be an ssh alias**, which is usually the least duplication. `address: dgx-a` picks up the user, port, key and `ProxyJump` from your `~/.ssh/config`. Peers inside a container on another machine cannot resolve an alias, and sflow does not read your config to guess one — set `rendezvous_address` to say what peers should dial.

Override either side only when they genuinely differ:

| Field | Overrides | Use when |
| --- | --- | --- |
| `docker_host` | the derived `ssh://` endpoint | a jump host, or a non-SSH endpoint (`tcp://…`) |
| `context` | the endpoint, via a Docker context | you already manage the daemon as a context |
| `rendezvous_address` | the peer-facing address | peers talk over a separate fabric (e.g. InfiniBand) |

`docker_host` and `context` are alternatives; setting both is rejected. A host needs either an `address` or one of those two overrides.

### 5. The fan-out is supervised, not scripted

Every member is driven from the driver rather than by a generated shell supervisor — which could only report a generic failure, could not name the rank that died, and had no way to label output per member.

- **Each member gets its identity**: `SFLOW_TASK_NODE_INDEX`, `SFLOW_TASK_NUM_NODES`, and `SFLOW_LEADER_ADDRESS`. Rank 0 is the first assigned host, deterministically.
- **One merged task log**, each line labelled with its node name and rank, so probes and post-run inspection stay on a single stream and probe config does not have to know the node count.
- **Fail fast**: the first member to exit non-zero fails the task **with its exit code**, and surviving peers are stopped immediately rather than burning GPUs on three other machines.
- **A multi-node task whose hosts cannot all supply a rendezvous address is rejected at planning time**, naming them — the silent-hang failure mode is gone.
- Environment values are forwarded with `-e KEY` so secrets are not embedded in the `docker run` command line.
- **Review the whole fan-out before launching.** `--dry-run --verbose` now prints one `docker run` per member, each labelled with its rank and the daemon it targets — so a misplaced member is visible at a glance instead of on the third read of a running job:

```
fan-out: 2 commands, one per member
   • node node-0 (rank 0) -> ssh://10.0.0.11
     docker --host ssh://10.0.0.11 run --rm --name sflow-...-node-0 ...
   • node node-1 (rank 1) -> ssh://10.0.0.12
     docker --host ssh://10.0.0.12 run --rm --name sflow-...-node-1 ...
```

### 6. Pre-flight the pool before anything launches

A backend with a declared `hosts:` pool checks each host once, up front:

- **the daemon answers** — a bounded `docker version` over the host's own endpoint, which exercises ssh auth, the tunnel, and the daemon in a single call;
- **bind-mount sources exist on that machine** — docker *creates* a missing bind source as an empty directory rather than failing, so a typo'd or unmounted path would otherwise become an empty mount and the workload would report missing data instead of a bad config.

Hosts are probed in parallel and **every problem across the pool is reported together, naming the host**, so a misconfigured pool takes one fix rather than one run per host. Path checks need a shell, so they apply to ssh endpoints; a `context:` host is skipped. Set `SFLOW_SKIP_DOCKER_PREFLIGHT=1` to bypass the host-touching checks.

### 7. Heterogeneous pools

A pool need not be uniform. Besides `gpus_per_node`, a host may carry its own `mounts` and `extra_args`, **added to** the backend-level ones for that host's containers only:

```yaml
    hosts:
      - name: dgx-a
        address: 10.0.0.11
        mounts:
          - /mnt/fast/models:/models:ro   # this box only
        extra_args: ["--shm-size=32g"]
```

Each member is pinned to a device slice computed for the host it lands on, so members do not silently share one slice. `include_nodes` / `exclude_nodes` filter the pool before placement, for pinning an experiment to specific machines or routing around a broken one.

### 8. Containers are never left behind

- **On normal completion, failure, signal, or Ctrl-C**, every member container is removed.
- **After a driver crash**, the next run reclaims the containers it left behind — on **every host**, not just the local daemon. The sweep is safe on a shared daemon: container names now carry the driver's *machine* identity as well as its PID, so a reaper can decide ownership from the name alone and can never touch a colleague's live containers.
- Teardown no longer stalls the driver. A reap against a *remote* daemon is an SSH round trip — seconds, not milliseconds — and running it inline froze the event loop for the duration, holding up every other task's probes and log pumps.

Set `SFLOW_DRIVER_ID` to a stable value when the driver itself runs **inside a container**: it otherwise gets a fresh machine-id per run and could never reclaim its own previous orphans — exactly the containerised-CI case the sweep is most needed for.

### 9. Hardware monitoring across a Docker pool

`monitor:` now covers multi-host `hosts:` pools. Those machines are not the driver, so a collector on the driver would report the *driver's* hardware under their names. Instead the collector runs **directly on each host** over ssh — bare metal, no container — and streams its samples back to the driver, which reassembles the on-disk layout the post-processor already reads.

Nothing is copied to the host and nothing is left behind: the collector's source is fed in on stdin.

- **The host needs `python3`** (standard library only) and, for the `gpu` scope, `nvidia-smi`. Nothing about the workload image matters: no image is pulled to sample, and an image whose entrypoint is not a shell (e.g. `vllm/vllm-openai`) no longer breaks monitoring. If the host has no `python3` the collector exits and **says so**; the run still succeeds, with no samples for that node.
- **Hosts with no ssh endpoint still use a container.** A `context:` or `tcp://` `docker_host:` names a daemon, not a machine sflow can log into, so there the collector runs as a container on that daemon from the backend's `image:` (which then needs `python3`). Its `gpu` scope follows `gpus_per_node`: requesting `--gpus` on a host with no NVIDIA runtime would fail the whole container, so a pool that declares no GPUs collects the other four scopes.
- **A collector that dies on its own now warns.** A collector that cannot start (no `python3`, no NVIDIA runtime, an unreachable host) previously left nothing above INFO anywhere, and the run finished "successfully" with an empty monitor report.
- **Collectors are stopped explicitly and concurrently at teardown.** Killing the local `ssh` does not stop the remote process, so each collector records its pid and is killed by it (a container is removed by name). Each stop is a round trip across the network, so serially this cost the sum of the pool in dead time after the workflow had already finished.

Kubernetes still does not support host monitoring — there is no DCGM/DaemonSet collector — and `monitor:` blocks are skipped there, unchanged.

---

## Improvements

- **Slurm walltimes no longer need quotes.** PyYAML follows YAML 1.1, which reads an unquoted `10:00:00` as the base-60 integer `36000`; Slurm then took it as 36000 *minutes*, so a 10-hour request became 25 days. sflow now loads configs with YAML 1.2 scalar rules, so `time: 10:00:00` and `time: "10:00:00"` are equivalent, and an unquoted `10:00:00.5` no longer fails validation. Only `H:MM:SS`-shaped values are affected. A bare integer (`time: 5400`) still means minutes, as in `sbatch --time`.
- **`version` is optional.** Omitting it means `"0.1"`, the only value that has ever existed. If declared it must still be `"0.1"`, and merged files must not disagree.
- **`sflow batch` directory scans pick recipes by their `workflow:` section** instead of a `version:` field. A recipe without `version` is now found, and a fragment such as a backends-only file is no longer submitted as a standalone recipe.
- **`sflow batch` reuses the running release correctly.** Without `--sflow-version`, a job installs the same sflow the submitting environment runs. For an install from PyPI or a private index, which records only a version, that version is now mapped to its release tag (`0.3.1` → `v0.3.1`). Previously the bare version was used as the ref, and the job's install failed.
- **Git installs work across the rename.** `sflow batch` and `sflow upgrade` install a git ref as a bare `git+<url>@<ref>`, so refs from before the rename (which build `sflow`) still install.
- **Docker auto-mounts local `fs://` / `file://` artifacts** as same-path bind mounts, so a task can read `${{ artifacts.X.path }}` without the recipe repeating the path under `mounts:`. srun has always done this; `docker_run` accepted the artifacts and ignored them, so the same recipe worked on Slurm and failed on Docker with a missing file.
- **Pre-flight narrates itself.** Each host-touching probe announces itself *before* the call, with its bound, so a pool that takes twenty seconds to validate cannot be mistaken for a hung driver.
- **Pre-flight is non-interactive.** The probes run with `BatchMode=yes` and no terminal, so an unknown host key or a passphrase-protected key fails immediately and attributably instead of putting sixteen parallel ssh prompts on one terminal.
- **Off-host backends no longer get bogus dry-run mount warnings.** Their mounts resolve on another machine, so stat-ing the controller reported paths "missing" that were present everywhere that mattered. The artifact-path warning also now says the paths must exist *on the execution hosts*, not on the controller.
- **The Docker backend's `image` is optional.** It is only used for tasks that declare no `operator:` (and for the container collector on a host with no ssh endpoint); a recipe whose tasks all name a `docker_run` operator — the usual shape once more than one image is in play — can leave it out. Either consumer fails with a message naming the field if it is reached without one.
- **`docker_run` gains `rm: false`** to keep a container after it exits for `docker logs` / `docker inspect`. Teardown then stops it instead of force-removing it; the next `sflow run` on that host still reaps it as an orphan, so inspect it first.
- **Offloaded task logs carry no ANSI escapes.** The `bash`/`srun`/`docker_run` offload prefixer strips colour and cursor codes at write time, and a Kubernetes merged-pod log is stripped live rather than only after the pod ends. `<task>.log` is read by regexes — `log_watch` probes, `results:`/`outputs:` parsing — and container logs are coloured even without a TTY (dynamo italicises every `key=value` field), so a readiness pattern such as `endpoint=generate` now matches while it matters.
- **New samples:** `sflow sample self_contained/docker/multi_node` (a two-host `torchrun` rendezvous smoke test — it prints two ranks only if the node-index and leader-address wiring is correct end to end), plus `self_contained/docker/dynamo_vllm_agg` and `self_contained/docker/dynamo_vllm_disagg` for aggregated and cross-node prefill/decode-disaggregated serving on a host pool. Three GLM recipes on Docker hosts: `self_contained/docker/sglang_glm52` (GLM-5.2 NVFP4, TP8 on one remote host, with the InferenceX AgentX benchmark), `self_contained/docker/sglang_disagg_glm53` and `self_contained/docker/vllm_disagg_glm53` (GLM-5.3-Flash prefill/decode disaggregation across a pool, pure SGLang and vLLM respectively).
- **`modular/backend_agnostic` gains a Slurm MPI fragment.** `backends/slurm_mpi.yaml` launches the server as one MPI rank per GPU under pmix (what TRT-LLM expects); `backends/slurm.yaml` stays the single-process form for vLLM/SGLang — the same split as `k8s.yaml` vs `k8s_mpi.yaml`. The model is now an `fs://` artifact declared on the backend fragment, so sflow mounts it into the container: override it with `-a LOCAL_MODEL_PATH=fs:///path` rather than `-s MODEL_PATH=...`. The TRT-LLM workload exposes `MAX_BATCH_SIZE` and `KV_FREE_GPU_MEMORY_FRACTION`, and the Kubernetes samples no longer carry the placeholder `node_selector: {tenant: your-node-tenant}` that had to be deleted before they would schedule.
- **Slurm dynamo vLLM samples run on current images and GB300.** `self_contained/slurm/dynamo_vllm_agg` and `dynamo_vllm_disagg` move from `vllm-runtime:0.8.0` to `vllm-runtime:1.2.1-cuda13` (0.8.0's Triton cannot target `sm_103a`), and the disagg sample uses the current `--disaggregation-mode prefill|decode` + NixlConnector `--kv-transfer-config` form instead of 0.8's `--is-prefill-worker`. Override `DYNAMO_IMAGE` if you pin something else.
- **Agent skills:** `check_gpu_plan.py` evaluates `${{ }}` arithmetic with a restricted AST evaluator instead of `eval()`, so a recipe can no longer run code through it. `preflight.py` and `validate_sflow_yaml.py` document their exit codes (`2` = usage error), `parse_sflow_errors.py` explains Jinja2 syntax errors more clearly, and the skill docs use the correct `${{ }}` syntax throughout.

---

## Breaking Changes

Ordered by how likely they are to affect an existing v0.3.1 setup.

1. **The distribution is renamed `sflow` → `nv-sflow`.** Tooling that names the distribution must use the new name: `pip show`, `pip uninstall`, requirement files, `importlib.metadata.version("sflow")`. GitHub builds keep the name `sflow`, so a `git+https://github.com/...` install still reports `sflow`. *Migration:* `pip install nv-sflow`, or `sflow upgrade` (twice from v0.3.1 or earlier); the leftover `sflow` record is removed on the next `sflow` run. Do **not** `pip uninstall sflow` before that: both names own the same files.
2. **Wheels up to and including v0.3.1 are named `sflow`.** On a private index, `--sflow-index-url <url> --sflow-version 0.3.1` now looks for `nv-sflow==0.3.1` and does not find it. *Migration:* install those releases from their git tag: `sflow batch --sflow-version v0.3.1` (no index URL) or `sflow upgrade --branch v0.3.1`.
3. **`sflow upgrade` defaults to the latest PyPI release**, not `main` of the public repo. *Migration:* `sflow upgrade --branch main` keeps the old behavior.
4. **`sflow upgrade --sflow-version` is a PyPI version only.** `--sflow-version develop` or `--sflow-version <repo>@<ref>` is now rejected, and the error points to the replacement. *Migration:* `--sflow-version <ref>` → `--branch <ref>`; `--sflow-version <repo>@<ref>` → `--repo <repo> --branch <ref>`. `sflow batch --sflow-version` is unchanged and still takes a git ref.
5. **Unquoted `H:MM:SS` values are now strings, everywhere in a config.** This is the fix above, but it applies to every field, not just `time`. A variable written as `DURATION: 1:30` used to load as the integer `90`; it now loads as the string `"1:30"`. *Migration:* write the number you mean (`DURATION: 90`).
6. **Docker `nodes: N>1` now requires a `hosts:` pool.** Without `hosts:` there is only this machine, and sflow used to invent `localhost-1` from it — advertising the box's full GPU count once per fake node, so the planner (correctly applying its one backend-agnostic rule) handed the same physical devices out twice. Multi-node docker means several *machines*. *Migration:* list the machines under `hosts:`, or set `nodes: 1` to run several containers on this one.
7. **Docker container names changed again** — `sflow-p<pid>-<task>-<node>` → `sflow-<machine>p<pid>-<task>-<node>`, where `<machine>` is a short hash of the driver's machine identity. A PID identifies an owner only *within* one machine, but a remote daemon is routinely shared by several drivers, so two machines could both produce `sflow-p1234-*` and a reaper checking its own `/proc` would judge a co-tenant's live container dead. *Migration:* update any script matching the old name. Containers left by a **pre-upgrade** sflow will not match the new sweep's filter and need one manual cleanup.
8. **Kubernetes dry-runs no longer emit container-mount path warnings.** Those mounts resolve inside the cluster, so stat-ing the controller reported paths "missing" that were present everywhere that mattered. *Migration:* none; if you relied on the warning, the paths were never being checked against the machine that actually matters.
9. **`ip_address` on a docker host is deprecated** in favour of `rendezvous_address` — or just `address`, when the machine is reachable at one address. It still works and now emits a warning naming the replacement. Setting `address` and `ip_address` to **different** values is honoured as the control/data split it describes: `address` is the endpoint the driver sends `docker run` to, `ip_address` the address peers dial. *Migration:* drop `ip_address` unless peers genuinely dial a different address, in which case rename it to `rendezvous_address`.
10. **Release tags must be in PEP 440 normal form** (maintainers only). The release job rejects `v0.4.0-rc1`. Tag `v0.4.0rc1` instead.

---

## Tested Environments & Current Support

**Validated setups**

- vanilla bare-metal Kubernetes
- Google Kubernetes Engine (GKE)
- Slurm (GRES and non-GRES partitions)
- a static pool of SSH-reachable Docker hosts (new in this release)

**Current limitations / work in progress**

- **Kubernetes hardware monitoring is still not supported** — no DCGM/DaemonSet collector; `monitor:` blocks are skipped on k8s targets. Unchanged since v0.3.0.
- **`--dry-run` renders command argv only for a Docker fan-out.** A multi-host task prints one `docker run` per member with the daemon each targets (see *Review the whole fan-out before launching* above). Other backends still show the plan — allocation, operator config, placement — without per-task argv.
- **There is no cross-host GPU lease.** The reservation registry is machine-local by design, so a task on a declared `hosts:` pool falls back to the planner's numeric device slice and warns. Own the hosts exclusively, or coordinate externally.
- **sflow orchestrates containers and supplies node metadata; it is not a distributed-compute scheduler.** Process ranks, NCCL/UCX transport and the launcher itself remain the workload's responsibility. Nothing queues, backfills or preempts — the pool is declared up front rather than discovered.
- **Controller-local paths are not auto-mounted on remote hosts.** A path that exists on the driver may not exist on the machine the container runs on; declare it under `mounts:` (or per-host `mounts:`).

---

## Documentation

- **[vLLM Recipes command builder](https://nvidia.github.io/nv-sflow/vllm-recipes)**: a docs-site page that assembles the `sflow` command for a vLLM recipe, including Docker bundles that work under any launcher and track the live nightly image tag. Builder fixes since the first cut: self-contained YAML downloads keep the namespace, PVC, local-model, cache, interface and Slurm settings entered on the page; copied commands and bundle scripts quote paths and values containing spaces or shell metacharacters; recipes without cache redirection disable that option, and read-only PVC cache destinations show the required volume change; failed asset requests show a Retry button; first open loads a gzip index (~16 kB instead of 286 kB); Docker/Kubernetes bundles for 39 single-node recipes no longer fail with `unknown flag: --export` (recipe env vars are exported in the server script instead of passed as Slurm-only `--export` arguments); DeepSeek-V4.1-Flash recipes use `vllm/vllm-openai:nightly`, replacing a pinned nightly tag removed from Docker Hub.
- **Project page** refreshed, with a Chinese version.
- **CLI → Run through SSH**: the full `--ssh` reference.
- **[Backends → Docker](https://nvidia.github.io/nv-sflow/docs/user/backends)**: the `hosts:` inventory, host identity and its overrides, pre-flight, reusing `~/.ssh/config`.
- **[Monitor](https://nvidia.github.io/nv-sflow/docs/user/monitor)**: monitoring a multi-host pool, and what each host needs (`python3`, `nvidia-smi`).
- **Outputs**: the Kubernetes log section now describes v0.3.1's actual behavior (streamed log, no post-run re-fetch).
- **Backends / Configuration / Quick Reference**: unquoted walltimes and optional `version`.
- New developer note: `docs/developer/dev-notes/docker-multi-host.md` — how one task spans several Docker hosts, why host identity collapsed to a single `address`, and what sflow deliberately does not do.

Full documentation: [nvidia.github.io/nv-sflow](https://nvidia.github.io/nv-sflow/)

---

## Upgrade Guide

```bash
# Either
pip install nv-sflow
# or (from v0.3.1 or earlier, twice: GitHub build first, then PyPI)
sflow upgrade

sflow --version     # the first run tidies up the old 'sflow' record

# Re-plan every recipe without consuming resources -- this surfaces the
# changes that fail at plan time (docker nodes > 1 without hosts:) and
# variables that relied on H:MM being read as a number
sflow run -f your_recipe.yaml --dry-run

# New samples worth a look
sflow sample self_contained/docker/multi_node
sflow sample self_contained/docker/dynamo_vllm_disagg
```

1. Replace `sflow @ git+...` install lines in scripts, Dockerfiles and requirement files with `nv-sflow`, or `nv-sflow==0.3.2` to pin.
2. Replace `sflow upgrade --sflow-version <ref>` with `sflow upgrade --branch <ref>`.
3. If you install pinned old releases from a private index, switch those to the git tag (Breaking Change 2).
4. `--dry-run` every recipe. A variable that relied on `H:MM` being read as a number now resolves to a string (Breaking Change 5). A Docker backend with `nodes: > 1` and no `hosts:` now fails at plan time — declare the machines under `hosts:`, or drop to `nodes: 1` (Breaking Change 6).
5. Grep your tooling for `sflow-p<pid>-` container names and update the pattern; sweep any containers left by a pre-upgrade sflow once by hand.
6. Moving to a host pool: declare each machine with one `address`, drop `docker_host`/`ip_address` unless they genuinely differ, and let pre-flight tell you what is unreachable.

Repository: [https://github.com/NVIDIA/nv-sflow](https://github.com/NVIDIA/nv-sflow)
