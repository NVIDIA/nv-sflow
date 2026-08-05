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

**Fix:** Add valid sflow YAML content to the file. At minimum: `version: "0.1"`.

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
- `version` not set to `"0.1"`
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

**Cause:** A `${{ }}` expression has invalid Jinja2 syntax.

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
- `${{ a + }}` -- incomplete expression
- `${{ variables.X | nonexistent_filter }}` -- unknown filter
- `${{ if x }}` -- Jinja2 uses `x if condition else y`, not `if condition`

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
to validate without it).

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

**Fix:** All files must use `version: "0.1"`.

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

**Cause:** After merging all files, required sections (version, workflow, tasks) are still missing.

**Fix:** Ensure the combined set of files provides all required sections. At minimum, one file must have `version` and `workflow.tasks`.

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
- For replicated tasks, ensure port calculation includes replica offset (e.g., `$((8082 + ${FIRST_CUDA_DEVICE}))`)

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
- Wrong cluster/context/namespace: pass `--kubeconfig`, `--kube-context`, `--kube-namespace`.
- Missing permission: grant the RBAC, or use a namespace where you have it.
- To bypass the check: `SFLOW_SKIP_K8S_PREFLIGHT=1`.

**Note on `Forbidden`:** a 403 means the API *was* reached and the credentials *did*
authenticate — it is an authorization gap, never a connectivity or kubeconfig problem. sflow
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

---

## 11. Docker Backend Errors

### 11.1 docker run failed / daemon unreachable

**Cause:** The Docker daemon isn't running/reachable, or `docker run` rejected the args.

**Fix:** Verify `docker info` works on the host; check the backend `image`, `mounts`,
`extra_args`. For remote hosts, verify the `docker_host` (`ssh://...`) or `context`.

### 11.2 NVIDIA container toolkit missing (GPU tasks)

**Cause:** `resources.gpus` requested but the host lacks the NVIDIA container toolkit.

**Fix:** Install the NVIDIA container toolkit on the Docker host and set `gpus_per_node` to
the host GPU count. sflow narrows `docker run --gpus device=...` to the planned slice.

### 11.3 Remote host path not found

**Cause:** For remote daemons (`docker_host`/`context`) sflow does not auto-mount local
workspace/output paths (they may not exist on the remote host).

**Fix:** Provide explicit backend/host/operator `mounts` pointing at a shared filesystem
available at the same path on every Docker host.

---

## 12. Storage / Upload Errors

### 12.1 boto3 Not Installed

**Pattern:**
```
S3 storage requires boto3. Install with: pip install 'sflow[s3]'
```
or (dry-run) `storage target '<name>' (s3) requires boto3, which is not installed`.

**Fix:** `pip install 'sflow[s3]'`.

### 12.2 No AWS Credentials

**Pattern (dry-run):**
```
storage target '<name>' (s3): no AWS credentials detected ...
```

**Cause:** No `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE` env and no
`~/.aws/credentials`. (An attached IAM role can't be detected offline.)

**Fix:** Provide credentials via the boto3 chain (env vars, `~/.aws/credentials`, or an
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
