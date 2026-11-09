# sflow Error Catalog

Comprehensive catalog of sflow error patterns organized by category. Each entry includes
the error text pattern, cause, fix, and an example where helpful.

> See also: [faq](https://nvidia.github.io/nv-sflow/docs/user/faq) for common questions and [cli](https://nvidia.github.io/nv-sflow/docs/user/cli) for CLI errors.

---

## 1. Config Loading Errors

### 1.1 File Not Found

**Pattern:**
```
✗ Configuration error: Configuration file not found: <path>
```
or
```
✗ File not found: <path>
```

**Cause:** The file path passed to `-f` or `--file` does not exist.

**Fix:** Verify the file path. Use absolute paths or paths relative to the current working directory.

---

### 1.2 Empty Configuration File

**Pattern:**
```
✗ Configuration error: Configuration file is empty: <path>
```

**Cause:** The YAML file exists but contains no content.

**Fix:** Add valid sflow YAML content to the file. At minimum: a `workflow:` section with a `name` and `tasks`.

---

### 1.3 YAML Syntax Error

**Pattern:**
```
✗ Configuration error: Error parsing YAML configuration: <yaml_error_detail>
```

**Cause:** Invalid YAML syntax (bad indentation, missing colons, unclosed quotes, tabs instead of spaces, etc.).

**Fix:** Check the line number in the error detail. Common issues:
- Mixed tabs and spaces (use spaces only)
- Missing `:` after a key
- Unclosed quotes or brackets
- Incorrect list indentation

**Example:**
```
Error parsing YAML configuration: while parsing a block mapping
  in "config.yaml", line 5, column 3
```
Fix: check line 5 for indentation or syntax issues.

---

### 1.4 Configuration Validation Failed

**Pattern:**
```
✗ Configuration error: Configuration validation failed: <pydantic_detail>
```

**Cause:** The YAML is syntactically valid but violates the sflow schema. Pydantic rejects unknown fields (`extra="forbid"`), wrong types, or constraint violations.

**Fix:** Read the Pydantic error detail carefully. Common issues:
- Extra/unknown fields in a section
- `version` declared with a value other than `"0.1"` (omitting it is fine)
- `gpus_per_node` not a positive integer
- `replicas.policy` not `"parallel"` or `"sequential"`
- `script` is empty or not a list
- Duplicate task names
- Invalid `depends_on` (referencing non-existent tasks)

---

### 1.5 Expression Syntax Validation Failed

**Pattern:**
```
✗ Configuration error: Configuration expression syntax validation failed:
  Expression: ${{ <expr> }}
  Error: <jinja2_error>
```

**Cause:** A ``${{ }}`` expression has invalid Jinja2 syntax.

**Fix:** Check the expression for:
- Unmatched brackets or parentheses
- Invalid filter names
- Python syntax not supported in Jinja2 sandbox
- Missing closing `}}`

---

### 1.6 Invalid Task Operator Override Key

**Pattern:**
```
Task '<task>': operator override key(s) not valid for operator '<op>' (type '<type>'): <keys>.
Check for a typo. If these are backend-specific operator settings, define them on the
'<op>' operator in the backend fragment (composed per backend via deep-merge) instead of
as a task-level operator override -- they are no longer silently dropped on an incompatible backend.
```

**Cause:** A task's inline `operator:` block carries a key that is **not a field of the
operator it resolves to** — a typo (e.g. `ntaks_per_node`), or a setting meant for a
different backend's operator (e.g. an srun `ntasks_per_node` on a `kubernetes` operator).
As of this change these keys are a **hard error**. (Earlier versions silently dropped
incompatible keys with a warning to keep a task portable across backends, but that hid
genuine typos — the misspelled key just vanished with no effect.)

**Fix:**
- If it's a typo, correct the key to a real field of that operator.
- If it's a genuinely backend-specific operator setting, move it off the task and onto the
  **operator definition inside the backend fragment**. The composer deep-merges operators
  by name per backend, so the setting applies only where that backend is used and the task
  stays portable. See the `writing-sflow-yaml` skill's schema-reference (composition /
  deep-merge) for the pattern.

---

## 2. Variable / Expression Errors

### 2.1 Undefined Variable in Expression

**Pattern:**
```
✗ Configuration error: Undefined variable in expression ${{ variables.TYPO_NAME }}: 'TYPO_NAME' is undefined
Source: line N in file.yaml
```

**Cause:** Expression references a variable not declared in `variables` or `workflow.variables`.

**Fix:** Check spelling. Declare the variable if it's new. Note that `--set` can only override existing variables, not create new ones.

---

### 2.2 Invalid Expression Syntax

**Pattern:**
```
✗ Configuration error: Invalid expression syntax in '<value>': <detail>
```

**Cause:** Jinja2 cannot parse the expression.

**Fix:** Common syntax issues:
- ``${{ a + }}`` -- incomplete expression
- ``${{ variables.X | nonexistent_filter }}`` -- unknown filter
- ``${{ if x }}`` -- Jinja2 uses `x if condition else y`, not `if condition`

---

### 2.3 Expression Evaluation Error

**Pattern:**
```
✗ Configuration error: Error evaluating expression ${{ <expr> }}: <detail>
```

**Cause:** Expression is syntactically valid but fails at evaluation (e.g., division by zero, type mismatch).

**Fix:** Check the expression logic. Ensure types are compatible (e.g., don't divide by a string).

---

### 2.4 Variable Override Not Defined

**Pattern:**
```
Variable '<key>' specified in overrides is not defined in the configuration.
```

**Cause:** Used `--set KEY=VALUE` but `KEY` is not declared in any YAML file.

**Fix:** Declare the variable in the YAML `variables` section first, then use `--set` to override.

---

### 2.5 Invalid Variable Override Format

**Pattern:**
```
Invalid variable override format: '<override>'. Expected KEY=VALUE.
```

**Cause:** The `--set` argument doesn't follow `KEY=VALUE` format.

**Fix:** Use `--set KEY=VALUE`. For lists: `--set CONCURRENCY=[16,32,64]`.

---

## 3. Artifact Errors

### 3.1 Artifact Path Does Not Exist

**Pattern:**
```
✗ Artifact path validation failed:
  Artifact '<name>' (fs://) path does not exist: <path>
```

**Cause:** An `fs://` artifact URI points to a path that doesn't exist on the filesystem. This
is raised only on a **real run** with a local/Slurm/Docker backend. On `--dry-run` a missing
`fs://` path is a **warning** (the run still validates), and on off-host backends (Kubernetes)
it also just warns (the path is treated as remote on the cluster/image).

**Fix:** Create the directory/file or fix the URI path before a real run (or use `--dry-run`
to validate without it). If the path is correct but exists only where the task runs (e.g. on
the Slurm compute nodes, not the submit host), pass `--skip-artifact-check` to `sflow run` /
`sflow batch` to demote this to a warning.

---

### 3.2 Artifact Override Not Defined

**Pattern:**
```
Artifact '<name>' specified in overrides is not defined in the configuration.
```

**Cause:** Used `--artifact NAME=URI` but `NAME` is not declared in the YAML.

**Fix:** Declare the artifact in the YAML `artifacts` section first.

---

### 3.3 Invalid Artifact Override Format

**Pattern:**
```
Invalid artifact override format: '<override>'. Expected NAME=URI.
```

**Cause:** The `--artifact` argument doesn't follow `NAME=URI` format.

**Fix:** Use `--artifact LOCAL_MODEL_PATH=fs:///path/to/model`.

---

### 3.4 Content with Non-file URI

**Pattern (Pydantic validation):**
```
ArtifactConfig: content is only valid with file:// URIs
```

**Cause:** Used `content:` field with a non-`file://` artifact URI.

**Fix:** Only `file://` URIs support inline content. For `fs://`, the file must already exist.

---

## 4. Merge / Composition Errors

### 4.1 Version Conflict

**Pattern:**
```
✗ Configuration error: Version conflict: '<a>' vs '<b>' (from <file_label>)
```

**Cause:** Two files being merged have different `version` values.

**Fix:** Make every file that declares `version` use `"0.1"`, or drop the field (it is optional and defaults to `"0.1"`).

---

### 4.2 Workflow Name / Duplicate Task Names (relaxed in v0.3.0)

**Behavior change:** As of v0.3.0 the multi-file merge is a recursive deep merge keyed on
`name`, so these are **no longer hard errors**:

- A differing `workflow.name` across files no longer raises `Workflow name conflict` — the
  **last non-null name wins** (a warning is logged).
- **Duplicate task names across files** are **deep-merged** into one task (a definition can be
  scattered across files) instead of raising `Duplicate task names`.

**Watch out:** if you relied on these as guardrails, a name collision now silently merges.
Audit composed files if a task looks like it picked up unexpected fields. (Duplicate task
names *within a single file* still raise — see 1.4.) Leaf scalars/value-lists still
last-file-wins with an override warning.

---

### 4.3 Incomplete Merged Configuration

**Pattern:**
```
✗ Configuration error: Merged configuration from [<files>] is incomplete: <detail>
```

**Cause:** After merging all files, required sections (`workflow`, its `name`, `tasks`) are still missing.

**Fix:** Ensure the combined set of files provides all required sections. At minimum, one file must have `workflow.name` and one must have `workflow.tasks`. `version` is optional.

---

## 5. SLURM Backend Errors

### 5.1 scontrol Failure

**Pattern:**
```
scontrol getaddrs failed with exit code <N>
```

**Cause:** sflow couldn't resolve node addresses via `scontrol`. Usually means the SLURM environment is not available.

**Fix:** Ensure you're running inside a SLURM allocation or on a node with `scontrol` access.

---

### 5.2 salloc Failure

**Pattern:**
```
salloc failed with exit code <N>. Output:
<salloc_output>
```

**Cause:** SLURM couldn't allocate nodes. Reasons include: invalid partition, invalid account, insufficient resources, reservation conflicts.

**Fix:**
- Check partition: `sinfo -p <partition>`
- Check account: `sacctmgr show account <name>`
- Check node availability: `sinfo -N -p <partition>`
- Reduce node count or wait for resources

---

### 5.3 sbatch Failure

**Pattern:**
```
sbatch failed: <stderr>
```

**Cause:** The generated sbatch script was rejected by SLURM.

**Fix:** Check the generated `.sh` script for invalid directives. Common issues:
- Invalid `--time` format
- Invalid `--partition` or `--account`
- `--gpus-per-node` exceeds available GPUs
- Invalid `extra_args`

---

### 5.4 GRES partition: the allocation has no GPUs

**Pattern (task log):**
```
nvidia-smi: No devices were found
RuntimeError: No CUDA GPUs are available
```
while the placement record `<task>/sflow_gpus.log` shows `visible_gpu_count=0` with
`nvidia_visible_devices_inherited=all`.

**Cause:** The partition declares `gpu:N` GRES and the allocation asked Slurm for none.
sflow's backend `gpus_per_node` is **planning-only** — it never adds `--gpus-per-node` to
`salloc`. On a non-GRES partition every step sees all of the node's GPUs anyway, which is why
the same recipe works there and not here.

**Fix:** Request the GPUs from Slurm: `--extra-salloc-args '--gpus-per-node=N'`, or
`extra_args: ["--gpus-per-node=N"]` on the backend.

---

### 5.5 Allocation revoked / nodes not ready (node boot failure)

**Pattern:**
```
salloc: error: Nodes <nodelist> are still not ready
salloc: error: Something is wrong with the boot of the nodes.
salloc: error: Job allocation <N> has been revoked.
```

**Cause:** The allocation was granted, then lost to a node prolog/boot failure; Slurm drains
the node (`sinfo -R` shows `Prolog error`). Not a recipe problem.

**Fix:** Re-run. If the same node keeps failing, route around it with
`--exclude-nodes <node>` until it is repaired.

---

### 5.6 Allocation expired while a partner backend was still queued

**Pattern:**
```
srun: error: Unable to confirm allocation for job <N>: Invalid job id specified
srun: error: Unable to confirm allocation for job <N>: Expired or invalid job <N>
```

**Cause:** In a two-backend recipe (e.g. `self_contained/slurm/multi_backend`) each backend
is its own `salloc`. One allocation's `time:` ran out while the other was still queued, so
when the tasks finally launched the first job no longer existed.

**Fix:** Raise `time:` on both backends (it must cover the partner's queue wait), run when the
partner partition is quieter, or put both on one backend.

---

## 6. Runtime / Task Errors

### 6.1 Task Python Traceback

**Pattern (in task log):**
```
Traceback (most recent call last):
  File "...", line N, in <module>
    ...
<ExceptionType>: <message>
```

**Cause:** The task script hit a Python exception. If a `failure.log_watch` probe is configured for `Traceback`, sflow will detect this and mark the task as failed.

**Fix:** Read the full traceback in the task log (`<task>/<task>.log`). Common causes:
- **ModuleNotFoundError**: Missing pip package in container
- **CUDA OOM**: Reduce batch size, TP/DP size, or use a smaller model
- **ConnectionRefusedError**: Upstream service (nats, etcd, frontend) not ready
- **FileNotFoundError**: Model path incorrect or not mounted in container

---

### 6.2 Probe Timeout

**Pattern (in sflow.log):**
```
Task '<task_name>' readiness probe timed out after <N>s
```

**Cause:** The task didn't emit the expected readiness pattern within the timeout period.

**Fix:**
- Check the task log for errors that prevented startup
- Verify `regex_pattern` matches what the server actually logs
- Increase `timeout` if the server takes longer to start
- For `tcp_port` probes: verify the port number matches the server config
- For `log_watch` with `logger`: verify the referenced task name is correct

---

### 6.3 Container Image Issues

**Pattern (in task log):**
```
slurmstepd: error: pyxis: ...
```
or
```
nvidia-container-cli: initialization error: ...
```

**Cause:** Container image couldn't be pulled or started. Pyxis/enroot issues.

**Fix:**
- Verify the container image URI is correct and accessible
- Check registry authentication
- Ensure enroot/Pyxis is properly configured on the cluster
- Try pulling the image manually: `enroot import docker://<image>`

**Variant — cold enroot cache race (first container start on a node):**
```
enroot-mount: failed to mount: /dev/nvidia-caps-imex-channels at <rootfs>/dev/nvidia-caps-imex-channels: No such file or directory
slurmstepd: error: pyxis: couldn't start container
```
or `mv: not replacing '<enroot cache path>'`. Several steps importing/unpacking the same
image into the node-local enroot cache at once; a plain retry succeeds. Pre-import the image
once per node before the run — `srun -N<n> --ntasks-per-node=1 --container-image=<img> true`
— or give the task `retries:`.

---

### 6.3b Image toolchain predates the GPU

**Pattern (task log):**
```
ptxas fatal   : Value 'sm_103a' is not defined for option 'gpu-name'
```
or `no kernel image is available for execution on the device`.

**Cause:** The container's CUDA/Triton is older than the GPU architecture (GB300 is
`sm_103a` and needs a CUDA 13 toolchain). `vllm-runtime:0.8.0` cannot target it;
`vllm-runtime:1.2.1-cuda13` can.

**Fix:** Use a newer image — `--set DYNAMO_IMAGE=<newer tag>` on the shipped samples.

---

### 6.3c aiperf measured nothing (exit 0)

**Pattern (task log):**
```
Processed 0 valid requests and <N> errors
```

**Cause:** Every request failed, but aiperf still exits 0, so the task is `COMPLETED` and
the CSV/JSON it wrote is empty. sflow has no result gate for this.

**Fix:** Read the error breakdown in the same log — HTTP 404 means the wrong `--endpoint` /
served model name, connection refused means the frontend died or the URL is wrong — then the
frontend/server task log. Always open the CSV before trusting a green benchmark.

---

### 6.4 Port Already in Use

**Pattern (in task log):**
```
OSError: [Errno 98] Address already in use
```

**Cause:** Another process is using the port the task is trying to bind to.

**Fix:**
- Kill stale processes from a previous run
- Use a different port
- For replicated tasks, ensure port calculation includes replica offset (e.g., ``$((8082 + ${FIRST_CUDA_DEVICE}))``)

---

### 6.5 NCCL / Communication Errors

**Pattern (in task log):**
```
NCCL error: ...
```
or
```
RuntimeError: NCCL communicator ...
```

**Cause:** Multi-GPU/multi-node communication failure. Often related to network config.

**Fix:**
- Set `NCCL_SOCKET_IFNAME` to the correct network interface
- Set `GLOO_SOCKET_IFNAME` similarly
- Check that all nodes can reach each other on the specified interface
- For InfiniBand: verify `NCCL_IB_HCA` settings

---

## 7. Batch / CSV Errors

### 7.1 Missing sflow_config_file Column

**Pattern:**
```
CSV file must contain a 'sflow_config_file' column.
```

**Cause:** The CSV file used with `--bulk-input` is missing the required column.

**Fix:** Add a `sflow_config_file` column with comma-separated YAML file paths.

---

### 7.2 Empty CSV

**Pattern:**
```
CSV file is empty: <path>
```
or
```
CSV file has no data rows: <path>
```

**Cause:** The CSV file has no rows (or only a header).

**Fix:** Add data rows to the CSV.

---

### 7.3 Unknown CSV Column

**Pattern:**
```
CSV column '<col>' is not a variable or artifact defined in any of the config file sets
```

**Cause:** A CSV column name doesn't match any declared variable or artifact.

**Fix:** Rename the column to match an existing variable/artifact name, or declare the variable in the YAML.

---

### 7.4 Missing Node Count

**Pattern:**
```
--nodes was not provided and the CSV does not contain a node-count column
```

**Cause:** Batch mode needs a node count but none was specified.

**Fix:** Either pass `--nodes N` on the command line or add a `SLURM_NODES` column to the CSV.

---

### 7.5 Dry-Run Validation Failed (Bulk)

**Pattern:**
```
============================================================
ERRORS: N config(s) failed dry-run validation:
============================================================
  [1] file1.yaml: <first line of error>
  [2] file2.yaml: <first line of error>
============================================================
```

**Cause:** One or more configurations in a bulk batch failed dry-run validation.

**Fix:** Fix each listed configuration individually. Run `sflow run -f <file> --dry-run` on each to see full error details.

---

## 8. CLI Argument Errors

### 8.1 Not Implemented: --resume

**Pattern:**
```
NotImplementedError: --resume is not implemented yet
```

**Cause:** `sflow run --resume` is a planned feature not yet available.

**Fix:** Re-run the full workflow instead.

---

### 8.2 Not Implemented: --task

**Pattern:**
```
Error: Bad parameter: Selective task execution (--task) is not yet implemented
```

**Cause:** `sflow run --task <name>` is a planned feature not yet available.

**Fix:** Run the full workflow. To skip tasks, restructure `depends_on` or use `--missable-tasks`.

---

### 8.3 Missable Tasks with Single File

**Pattern:**
```
Error: --missable-tasks is only valid with multiple input files (modular configs).
```

**Cause:** Used `--missable-tasks` with a single `-f` file. It only works with multi-file composition.

**Fix:** Either split the config into multiple files or remove the `--missable-tasks` flag.

---

### 8.4 --row without --bulk-input

**Pattern:**
```
Error: --row requires --bulk-input.
```

**Cause:** Used `--row` flag without specifying `--bulk-input`.

**Fix:** Add `--bulk-input <csv_file>` to the command.

---

### 8.5 Invalid Slice for --row

**Pattern:**
```
Invalid slice: '<token>'
```
or
```
Slice step cannot be zero
```

**Cause:** Invalid Python-style slice syntax in `--row`.

**Fix:** Use valid slice syntax: `--row 0` (single row), `--row 0:5` (range), `--row 0:10:2` (step).

---

## 9. Visualization Errors

### 9.1 Graphviz Not Found

**Pattern:**
```
dot failed with exit code <N>
```
or system cannot find `dot`.

**Cause:** Graphviz is not installed but `--format png/svg/pdf` was requested.

**Fix:** Install Graphviz (`apt install graphviz` or `conda install graphviz`), or use `--format mermaid` or `--format dot` which don't require Graphviz.

---

## 10. Kubernetes Backend Errors

The kubernetes backend runs `kubectl` on the `sflow run` host. Most failures surface either
in a fast pre-flight (before allocation) or as pod status. Use `kubectl -n <ns> describe pod
<pod>` and `kubectl -n <ns> get events` to inspect.

### 10.1 Access Pre-flight Failure (RBAC / namespace / unreachable)

**Cause:** On a real run sflow verifies the cluster is reachable and authenticated, the
namespace exists, and (via `kubectl auth can-i`) that the credentials hold the RBAC it needs.
Only **namespaced** permissions it depends on hard-fail: create/delete pods, configmaps,
secrets; get pods/logs; DRA `resourceclaimtemplates`; `computedomains` when creating one;
`mpijobs` on the `k8s_mpi` operator route. Denied **cluster-scoped** reads (`namespaces`,
`nodes`, `deviceclasses`) and a forbidden `auth can-i` only warn.

**Fix:**
- Wrong cluster/context/namespace: pass `--kube-context` / `--kube-namespace` (or the config-file
  override in `sflow run --help`).
- Missing permission: grant the RBAC, or use a namespace where you have it.
- To bypass the check: `SFLOW_SKIP_K8S_PREFLIGHT=1`.

**Note on `Forbidden`:** a 403 means the API *was* reached and the credentials *did*
authenticate — it is an authorization gap, never a connectivity or cluster-config problem. sflow
no longer fails on 403s for permissions it does not use, so
`namespaces "<ns>" is forbidden: User "system:serviceaccount:..." cannot get resource
"namespaces"` — the normal result for a namespaced `Role` on a shared cluster — is only a debug
note. If you still see it as a hard error, you are on an older sflow: bypass with
`SFLOW_SKIP_K8S_PREFLIGHT=1`. When node reads are denied, set `gpus_per_node` explicitly (sflow
cannot derive it from node capacity).

### 10.1b `kubectl apply` rejected (validation / field manager / conflicts)

**Pattern:** `error validating data: ...`, `Apply failed with N conflicts`, or an admission
rejection on apply — at allocate (`Failed to create reservation Pod '...'`) or in a task's
apply step.

**Cause:** the cluster needs an extra flag on the `apply` subcommand (`--validate=false`,
`--server-side`, `--force-conflicts`, `--field-manager=...`).

**Fix:** pass it with `--extra-kubectl-apply-args` (repeatable), which applies to every apply
sflow issues — allocate-time reservation/ComputeDomain/ResourceClaimTemplate *and* task pods:

```bash
sflow run -f recipe.yaml --extra-kubectl-apply-args=--validate=false
```

**Do NOT use `--extra-kubectl-args` for these.** That option carries *global* flags, which
kubectl requires **before** the subcommand — `kubectl --validate=false apply` fails with
`unknown flag`, and because the flag is prefixed onto *every* call it breaks `get`/`logs`/
`delete` too, usually surfacing as a confusing pre-flight or allocation error far from the
typo. sflow warns when it detects an apply-only flag in the global option.

### 10.2 Pod Stuck `Pending` (unschedulable)

**Pattern (`kubectl describe pod`):** `0/N nodes are available: insufficient nvidia.com/gpu`
or no node matches the nodeSelector/affinity.

**Cause:** No node can satisfy the GPU request or the node-selector/tolerations/`include_nodes`.

**Fix:** Check `gpus_per_node` vs real capacity, the `scheduling` mode (`dra` needs
`nvidia-dra-driver-gpu`; `device_plugin` needs the gpu-operator), `node_selector` /
`tolerations`, and that `resources.gpus.count` is a multiple of the node count.

### 10.3 `ImagePullBackOff` / `ErrImagePull`

**Cause:** The `k8s` operator `image` is wrong or private without credentials.

**Fix:** Fix the operator `image`; for private registries add `image_pull_secrets: [<secret>]`
(the workload image lives on the **operator**, not the backend).

### 10.4 GPU count not divisible by node count

**Cause:** For a multi-node k8s task, `resources.gpus.count` is split evenly across nodes.

**Fix:** Make `gpus.count` a multiple of the assigned node count (or reduce nodes).

### 10.5 kubectl flag not recognized

**Pattern:** an `--extra-args` value that is not a valid global kubectl flag is applied to
every `kubectl` call and emits a warning (or kubectl rejects it).

**Fix:** Use `--extra-kubectl-args` for kubectl-only flags (or `--extra-args`, which is generic
across Slurm/docker/kubectl). Don't pass Slurm-isms (e.g. `--gpus-per-node`) as kubectl flags.

### 10.6 Readiness timeout after an OOM right after another run's pods were deleted

**Pattern:** `Task '<task>' readiness probe timed out: ...` in `sflow.log`, and shortly
before it in the task log a TRT-LLM `OOM during general warmup` or
`ncclMemAlloc ... unhandled cuda error`, on a node where another recipe's pods were deleted
moments earlier.

**Cause:** GPU memory from the previous pods had not been released yet when the new server
started warming up. Not a recipe problem; the same recipe passes on a cold node.

**Fix:** Wait a minute or two (or check `nvidia-smi` on the node) before relaunching on the
same node, or pin to a different node with `--include-nodes`.

---

## 11. Docker Backend Errors

### 11.0 Plan-time rejections (before anything launches)

| Message | Fix |
|---------|-----|
| `Docker backend '<b>' asks for <N> nodes but declares no 'hosts:'. Multi-node docker runs one container per machine, so list the machines under 'hosts:' -- or set 'nodes: 1' to run several containers on this one.` | Several machines → declare them under `hosts:`; one machine → `nodes: 1` |
| `Docker backend '<b>' declares <H> host(s) but asks for nodes: <N>. 'hosts:' is the node list, so the run would use <H> and ignore the rest. Drop 'nodes:', or declare a host per node.` | Remove `nodes:` or make it equal the pool size |
| `docker host '<h>' needs an 'address' (the machine), or a 'docker_host'/'context' endpoint override` | Add `address: <ip/hostname/ssh alias>` |
| `docker host '<h>' has address '<v>': drop the scheme; 'address' is a host, not a URL` / `put the login in 'user' instead` / `an address cannot contain whitespace` | `address` is a bare host; put `ssh://` in `docker_host` and the login in `user` |
| `docker host '<h>' sets both 'docker_host' and 'context'; they are alternative endpoint overrides, so set at most one` | Keep one |
| `docker host '<h>' sets 'user' without 'address'. 'user' decorates the endpoint derived from 'address'; with an explicit 'docker_host' put the user in the URL instead` | `docker_host: ssh://user@host`, or switch to `address` + `user` |
| `Docker backend '<b>': multi-node task cannot resolve a rendezvous address for host(s) <names>. Give each host an 'address' (or 'rendezvous_address'); peers need it to find the leader.` | A `context:`-only host has no machine address; give it `address` or `rendezvous_address` |
| `Task '<t>' declares no 'operator:' and Docker backend '<b>' has no 'image:' to run it with -- set one of them.` | Name a `docker_run` operator on the task, or set the backend `image` |
| `Docker backend '<b>': host '<h>' has no ssh endpoint, so its monitor collector runs as a container and needs the backend's 'image:' -- set it, or give the host an 'address'.` | `monitor:` on a `context:`/`tcp://` host runs in a container from the backend `image` |
| `Docker backend '<b>': include/exclude node filters removed all hosts from the pool` | Fix `--include-nodes` / `--exclude-nodes` (they match `hosts[].name`) |
| *warning* `docker host '<h>': 'ip_address' is deprecated, use 'rendezvous_address' (or just 'address' when the machine is reachable at one address)` | Rename it, or drop it when `address` is what peers dial. An `address` and `ip_address` that disagree are honoured as a control/data split |

### 11.1 Pre-flight failed / daemon unreachable

**Pattern:** a declared `hosts:` pool is probed before launch (in parallel, 20 s bound per
probe) and every problem is reported together:
```
Pre-flight validation failed for Docker backend '<b>':
  - host '<h>' (ssh://user@10.0.0.11) unreachable: <last line of ssh/docker stderr>
  - host '<h>' (ssh://dgx-b): no response within 20s
  - host '<h>': mount source(s) not found on that machine: /mnt/models, /data
```
or, with no docker CLI on the driver at all,
`Pre-flight validation failed for Docker backend '<b>'. Missing required command: docker.`
The narrated INFO lines (`Pre-flight: validating N declared Docker host(s) ...`,
`Pre-flight host '<h>': daemon OK (server 27.x)`, `... mount sources OK`) tell you how far
it got; the warning `Pre-flight host '<h>': artifact path(s) not present yet: ... Docker will
create them empty` is informational (a task is expected to populate them).

**Cause / fix per line:**
- `unreachable: <reason>` — the reason is ssh's or docker's last stderr line. Probes run with
  `BatchMode=yes` and no terminal, so `Host key verification failed` = unknown host key (ssh
  once by hand), `Permission denied` = key/agent/passphrase not usable non-interactively, a
  `docker:` error = daemon down or user not in the `docker` group on that machine. Verify
  `ssh <address> docker version` works from the driver without a prompt.
- `no response within 20s` — network/DNS/ProxyJump; check `~/.ssh/config` for that alias.
- `mount source(s) not found on that machine` — a declared `mounts:` source (backend, host
  or operator) or a local `fs://` artifact path does not exist **on that host**. Docker would
  create it empty and the workload would report missing data, so sflow stops here. Create it,
  fix the path, or give that host its own `mounts:`.
- Path checks need a shell, so a `context:` host is skipped; `docker_host`/`context`
  overrides are probed as written.
- `SFLOW_SKIP_DOCKER_PREFLIGHT=1` bypasses the host-touching checks (not the `docker` CLI check).

A plain `Cannot connect to the Docker daemon` on a run without `hosts:` means the local
daemon: `docker info` on the driver.

### 11.2 NVIDIA container toolkit missing (GPU tasks)

**Cause:** `resources.gpus` requested but the host lacks the NVIDIA container toolkit.

**Fix:** Install the NVIDIA container toolkit on the Docker host and set `gpus_per_node` to
the host GPU count. sflow narrows `docker run --gpus device=...` to the planned slice.

### 11.3 Remote host path not found

**Cause:** On a declared `hosts:` entry sflow does not auto-mount the controller's
workspace/output paths (they may not exist on that machine). Local `fs://`/`file://`
artifacts *are* bind-mounted at their own path, but pre-flight fails the run when that
path is missing on a host — docker would otherwise create it as an empty directory.

**Fix:** Provide explicit backend/host/operator `mounts` pointing at a shared filesystem
available at the same path on every Docker host, or per-host `mounts` for the one that
differs; make artifact paths exist on each host.

### 11.4 Leftover containers / `docker ps` shows `sflow-…` after a run

Containers are named `sflow-<machine>p<pid>-<task>-<node>` (`<machine>` is a short hash of
the driver's machine identity) and are removed on completion, failure, signal and Ctrl-C.
After a driver crash the **next** run reclaims its own orphans on every host it targets,
never a colleague's live ones.

- Containers named `sflow-p<pid>-…` or `sflow-<task>-<node>` were left by a **pre-v0.3.2**
  sflow; the sweep's filter does not match them — clean them up once by hand.
- A driver running **inside a container** (CI) gets a fresh machine identity per run and
  cannot recognise its previous orphans: set `SFLOW_DRIVER_ID` to a stable value.
- A container kept with `docker_run.rm: false` is stopped, not removed, at teardown and is
  reaped as an orphan by the next `sflow run` on that host — `docker logs` / `inspect` it first.
- A multi-node member that exited non-zero fails the whole task with its exit code and the
  peers are stopped at once; look for the rank label in the merged task log to find which
  member died.

---

## 12. Storage / Upload Errors

### 12.1 boto3 Not Installed

**Pattern:**
```
S3 storage requires boto3. Install with: pip install 'nv-sflow[s3]'
```
or (dry-run) `storage target '<name>' (s3) requires boto3, which is not installed`.

**Fix:** `pip install 'nv-sflow[s3]'`.

### 12.2 No AWS Credentials

**Pattern (dry-run):**
```
storage target '<name>' (s3): no AWS credentials detected ...
```

**Cause:** No `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE` env and no
AWS shared credentials file. (An attached IAM role can't be detected offline.)

**Fix:** Provide credentials via the boto3 chain (env vars, the AWS shared credentials file, or an
instance/pod IAM role). **Never put secrets in YAML.**

### 12.3 Unknown Storage Target

**Pattern:**
```
Task '<name>' upload references unknown storage target '<target>'. Declared targets: ...
```
or `workflow.upload_all references unknown storage target '<target>'`.

**Fix:** Declare a matching entry under the top-level `storage:` block, or fix the `target:`.

### 12.4 Glob `to` Must End With `/`

**Pattern:**
```
upload 'to' must end with '/' when 'from' contains a glob pattern ...
```

**Cause:** A literal `to:` against a glob `from:` is ambiguous across runs.

**Fix:** End `to:` with `/` (directory) or omit it.

### 12.5 Upload Failure at Runtime

**Cause:** Bucket/region/endpoint wrong, network, or permissions. Uploads are **not** retried
by the task's `retries:` block.

**Fix:** For a hard failure, set `on_error: fail` only when the upload IS the deliverable;
otherwise `warn` (default) keeps the task `COMPLETED`. Verify `bucket`/`region`/`endpoint_url`.

---

## 13. Schema Validation Errors

### 13.1 Operator `type: docker` (use `docker_run`)

**Pattern:**
```
Operator type 'docker' is not valid; use `type: docker_run` for the Docker launch operator.
```

**Fix:** Use `type: docker_run` for the operator. The Docker **backend** is `type: docker`, but its launch operator is `type: docker_run`.

### 13.2 Result Parsing Validation

**Patterns:**
```
result.patterns and result.file are mutually exclusive in this release
result.file must point to a JSON source path ending in '.json'; for a metric named 'file', use result.patterns
```

**Fix:** Use either `patterns:` or `file:` (not both); `result.file` must end in `.json`.

### 13.3 Monitor Validation

**Patterns:**
```
monitor interval must be >= 100ms (got <N>); ...
monitor.scopes is set but no scope is active; enable at least one built-in scope (...) or add scopes.custom
workflow.monitor.resources.used_by_tasks refers to unknown task '<ref>'
```

**Fix:** Use an `interval` >= 100 (ms); enable at least one scope when `scopes:` is present;
ensure `used_by_tasks` names existing tasks.

---

## 14. Install / Upgrade Errors

The distribution is **`nv-sflow`** on PyPI from v0.3.2 (GitHub builds and v0.3.1 and
earlier are named `sflow`; the import name and CLI are unchanged). `sflow upgrade` installs
the latest PyPI release by default; `--sflow-version` is a PEP 440 version/specifier only;
`--branch` / `--repo` install a git ref.

### 14.1 `--sflow-version` given a git ref

**Pattern:**
```
Error: --sflow-version 'develop' is not a valid PyPI version specifier. Use a plain version like '0.2.1' or a specifier like '>=0.2,<0.3'; package names, URLs and '@' direct references are not allowed. For a git branch or tag use --branch/--repo.
```

**Fix:** `sflow upgrade --sflow-version <ref>` → `sflow upgrade --branch <ref>`;
`--sflow-version <repo>@<ref>` → `--repo <repo> --branch <ref>`. (`sflow batch
--sflow-version` is unchanged and still takes a git ref unless `--sflow-index-url` is set.)

### 14.2 Two sources at once

**Pattern:** `--repo/--branch install a git ref and cannot be combined with --sflow-version, --sflow-index-url; a PyPI version or index does not apply to a git checkout.` or
`--sflow-source-path is mutually exclusive with ...`.

**Fix:** Pick one route: PyPI (`--sflow-version` / `--sflow-index-url`), git
(`--repo` / `--branch`), or a local checkout (`--sflow-source-path`).

### 14.3 Refusing to replace a development install

**Pattern:** `Error: sflow is currently a 'editable' (development) install (<path>). Upgrading would replace it with a released build and your local changes would no longer be used. Re-run with --force if that is what you want, or reinstall from your checkout with 'pip install -e .'.`

**Fix:** That is the guard working. `--force` to replace the checkout with a release;
otherwise `git pull` in the checkout instead.

### 14.4 No installer

**Pattern:** `Error: neither 'uv' nor 'pip' is available in this environment, so sflow cannot install anything. Install uv (https://docs.astral.sh/uv/) or add pip with '<python> -m ensurepip --upgrade', then retry.`

**Fix:** As the message says — uv-created venvs routinely ship without pip.

### 14.5 Installer failed / old release not on a private index

**Pattern:** `Error: install failed with exit code <N>.` after the installer's own output,
e.g. `No solution found ... nv-sflow==0.3.1` / `No matching distribution found for nv-sflow==0.3.1`.

**Cause:** Wheels up to and including v0.3.1 were published under the old name `sflow`, so
`--sflow-index-url <url> --sflow-version 0.3.1` looks for `nv-sflow==0.3.1` and finds nothing.

**Fix:** Install old releases from their git tag: `sflow upgrade --branch v0.3.1`, or
`sflow batch --sflow-version v0.3.1` (no index URL). Otherwise read the installer output above
the error line (network, index auth via `~/.netrc`, Python floor 3.10).

### 14.6 `sflow` broken after `pip uninstall sflow` / wrong package installed

**Cause:** `sflow` and `nv-sflow` own the same files, so `pip uninstall sflow` after an
upgrade deletes the working install. `pip install sflow` installs an **unrelated** project
from pypi.org.

**Fix:** `pip install --force-reinstall nv-sflow`. Upgrading from v0.3.1 or earlier: `pip
install nv-sflow`, or `sflow upgrade` **twice** (the old command lands on the GitHub build,
which then moves you to PyPI); the stale `sflow` record is tidied on the next `sflow` run
(`SFLOW_NO_AUTO_MIGRATE=1` disables that).

### 14.7 Credentials in the index URL

**Pattern:** `--sflow-index-url must not contain embedded credentials; use ~/.netrc or a credential helper instead.` (also rejects query strings / fragments).

**Fix:** Put credentials in `~/.netrc` or a credential helper; the URL is embedded verbatim
in generated scripts and logs.
