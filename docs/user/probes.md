---
title: Probes
sidebar_position: 8
---

Probes let you gate task execution on an external condition, like:

- “wait until a TCP port is open”
- “wait until an HTTP endpoint returns success”
- “wait until a log line appears”
- “fail the workflow early when an error pattern appears”

You can use probes under:

- `probes.readiness`: wait before treating the task as ready (so dependents can run)
- `probes.failure`: mark task as failed early if a failure condition is met

Common timing options:

| Option | Default | Applies to | Meaning |
|--------|---------|------------|---------|
| `delay` | `0` | both | Seconds to wait before the first check. |
| `interval` | `5` | both | Seconds between checks. |
| `each_check_timeout` | `30` | both | Per-check timeout in seconds (e.g. how long one TCP connect / HTTP request may take). |
| `timeout` | `1200` | readiness | Overall readiness deadline in seconds. Only **readiness** probes time out the task; failure probes keep checking for the task's lifetime. |
| `success_threshold` | `1` | readiness | Consecutive successful checks required before the probe is satisfied. |
| `failure_threshold` | `3` | failure | Consecutive matching checks required before the probe fails the task. |

Both `readiness` and `failure` may be a single probe or a list of probes. When multiple readiness probes are configured, the task becomes ready only after every readiness probe has triggered; when multiple failure probes are configured, any one of them can fail the task.

## Readiness: TCP port probe

Example:

```yaml
version: "0.1"

workflow:
  name: http_echo
  tasks:
    - name: echo_server
      script:
        - python3 -m http.server 8000
      probes:
        readiness:
          tcp_port:
            port: 8000
          timeout: 30
          interval: 1
    - name: echo_client
      depends_on: [echo_server]
      script:
        - curl -sf http://127.0.0.1:8000/ > /dev/null
```

`tcp_port` fields:

- `port` (required): the TCP port to connect to.
- `host`: host or IP to probe (defaults to the task's assigned node).
- `on_node`: `"first"` (default) checks only the first assigned node; `"each"` checks the port on every node assigned to the task.

```mermaid
flowchart TD
  echo_server[echo_server] -->|readiness: tcp_port 8000| ready{{READY}}
  ready --> echo_client[echo_client]
```

## Readiness: HTTP probes

Use `http_get` or `http_post` when an HTTP endpoint is a better health signal than an open port:

```yaml
workflow:
  name: http_ready
  tasks:
    - name: api_server
      script:
        - python -m my_server --port 8000
      probes:
        readiness:
          http_get:
            url: "http://127.0.0.1:8000/health"
            headers:
              Accept: application/json
          timeout: 120
          interval: 2
    - name: client
      depends_on: [api_server]
      script:
        - curl -sf http://127.0.0.1:8000/health
```

A check counts as **success when the endpoint returns any `2xx` or `3xx` status** (the
range `200–399`, so redirects like `301`/`302` also pass). `4xx`/`5xx` and connection
errors count as not-ready.

`http_post` supports the same `url` and `headers` fields plus an optional `body`:

```yaml
probes:
  readiness:
    http_post:
      url: "http://127.0.0.1:8000/v1/health"
      headers:
        Content-Type: application/json
      body: '{"ping": true}'
```

> `body` applies only to `http_post`. If you set it under `http_get`, it is ignored.

> **Kubernetes:** TCP/HTTP probes normally run from the `sflow run` host. On the Kubernetes
> backend they run **from inside the cluster** (via a small per-allocation probe pod) so they
> still work when the driver host cannot reach the pod network. This is automatic — see
> [Readiness probes run in-cluster](backends.md#readiness-probes-run-in-cluster-probe-pod).

## Readiness: log watch probe (+ retries)

`log_watch` scans a task's log file for a matching string.

**Pattern field** — use one of (not both):

| Field | Description |
|-------|-------------|
| `regex_pattern` | Original field name |
| `match_pattern` | Alias (identical behavior, for forward compatibility) |

**Matching behavior:**

- By default the pattern is treated as a **literal string match** — characters like `(`, `)`, `.`, `*` are matched as-is, no escaping needed.
- To use a real regex, prefix the pattern with `re:` (or `regex:`).

| Pattern value | What it matches |
|---------------|-----------------|
| `"server started"` | Literal text `server started` |
| `"Traceback (most recent call last)"` | Literal text including the parentheses |
| `"re:worker_\\d+ ready"` | Regex: `worker_` followed by one or more digits, then ` ready` |
| `"regex:ERROR\|FATAL"` | Regex: `ERROR` or `FATAL` |

**Other options:**

- `logger`: watch another task's log instead of the current task's (must be a valid task name)
- `match_count`: number of times the pattern must appear before the probe passes (default `1`; accepts `${{ }}` expressions)

```yaml
workflow:
  name: wf
  tasks:
    - name: worker
      script:
        - echo "Setting PyTorch memory fraction"
        - sleep 999
      probes:
        readiness:
          log_watch:
            regex_pattern: "Setting PyTorch memory fraction"
          timeout: 600
          interval: 10
      retries:
        count: 3
        interval: 10
        backoff: 2
```

```mermaid
flowchart TD
  worker[worker] -->|readiness: log_watch| ready{{READY}}
```

## Failure probes

Failure probes watch for conditions that should stop the workflow early. A common pattern is to watch long-running server logs for tracebacks or fatal errors:

```yaml
workflow:
  name: wf
  tasks:
    - name: server
      script:
        - start_server.sh
      probes:
        readiness:
          log_watch:
            match_pattern: "server ready"
          timeout: 600
        failure:
          log_watch:
            match_pattern: "Traceback (most recent call last)"
            match_count: 1
          interval: 2
          failure_threshold: 1
    - name: benchmark
      depends_on: [server]
      script:
        - run_benchmark.sh
```

When a failure probe triggers, sflow marks the task as failed by probe and cancels downstream work through fail-fast. Failure probes do not use the overall `timeout` as a deadline; they keep checking while the task is running. `each_check_timeout` still applies to each individual check.

### Debugging a probe that never triggers

The **Probe Traces (last attempt)** section of [`sflow_summary.log`](./outputs.md#execution-summary) shows what each probe last observed — for a `log_watch` probe the matched line (or the last line it saw on a miss) and its running match count, for `tcp_port`/`http_get`/`http_post` the endpoint and result. It refreshes live while the task is RUNNING, so if a readiness probe hangs it's the fastest way to tell whether the marker never appeared or the pattern simply didn't match (`log_watch` is literal unless prefixed `re:`/`regex:`).

## Replicas and HTTP probe deduplication

For parallel replicas, identical HTTP probes that do not reference per-replica values are checked once on the first replica and propagated to follower replicas. This avoids sending the same health check N times when all replicas share one service endpoint.

sflow keeps a separate HTTP probe on every replica when the probe references a per-replica value such as:

- a swept variable from `replicas.variables`
- `SFLOW_REPLICA_INDEX`

TCP probes always stay per replica because each replica may expose a different port or node binding.
