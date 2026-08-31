# sflow GPU placement -- sourced (never executed) by each Slurm job step.
#
# MUST be sourced: it exports CUDA_VISIBLE_DEVICES into the task's own shell, and
# a subprocess could not.
#
# Inputs (environment):
#   SFLOW_GPU_PLAN            planned HOST device indices, e.g. "2,3"
#   SFLOW_PLANNED_GPU_UUIDS   "<node>=<uuid>,<uuid>;..." from the driver's
#                             bare-metal probe; absent => fall back to the old
#                             index arithmetic
#   SFLOW_GPU_MARKER          record filename under SFLOW_TASK_OUTPUT_DIR
#
# It answers ONE question, the same way every time: which indices do the planned
# cards have HERE? It probes the devices this step can actually see, looks each
# planned UUID up among them, and names the indices it found. That single rule
# covers every shape -- more devices visible than planned (narrow to them),
# exactly as many (fix a stale value naming the wrong ordinals), a container that
# renumbered from 0 (just a different lookup result), nothing set at all (name
# them explicitly, because recipes deref this under `set -u`). There is no
# separate no-op / pin / narrow mode, because those were all this one question.
#
# UUID is the identity, never the index: an index stops meaning anything the
# moment a layer renumbers. A planned card that is not visible AT ALL is a hard
# error (exit 97) when sflow chose the devices, and a degrade to index arithmetic
# when Slurm did -- GRES may have granted cards the planner never saw, and there
# the plan can only mean a position into the grant.
#
# Where the UUIDs cannot be checked at all (no map for this node, or no nvidia-smi
# here) it degrades to that same index arithmetic rather than stopping. Narrowing
# a step handed the whole allocation is the collision this exists to prevent, and
# a KNOWN plan must never end up doing less than an unknown one.
#
# It leaves an audit record next to the task's logs: what was planned, what the
# step could see (index -> UUID), what CUDA ended up selecting, and what arrived
# beforehand -- inherited equal to final IS the "nothing needed changing" signal.
# Compared with the Node Topology section of the run summary, that is what tells
# a bad placement apart from a recipe using the wrong device.
[ -n "${SFLOW_GPU_PLAN:-}" ] || return 0

__sflow_plan="${SFLOW_GPU_PLAN:-}"
__sflow_node="${SLURMD_NODENAME:-$(hostname -s)}"
# Physical GPUs planned for THIS node; empty => fall back to index arithmetic.
__sflow_want=""
if [ -n "${SFLOW_PLANNED_GPU_UUIDS:-}" ]; then
  __sflow_oifs="$IFS"; IFS=';'
  for __sflow_e in ${SFLOW_PLANNED_GPU_UUIDS}; do
    case "$__sflow_e" in
      "$__sflow_node="*) __sflow_want="${__sflow_e#*=}" ;;
    esac
  done
  IFS="$__sflow_oifs"
  # A map that names other nodes but not this one means the driver's node names
  # and $SLURMD_NODENAME disagree (short vs FQDN is the usual cause). Verification
  # silently switches off there, so say it once.
  [ -n "$__sflow_want" ] || echo "sflow: no planned-GPU entry for node '$__sflow_node' in SFLOW_PLANNED_GPU_UUIDS; falling back to device-index placement" >&2
fi

# What arrived in the step, before this script touches anything. Without it the
# record shows only the post-state and a reader cannot tell what changed or why.
#
# BOTH variables, because they answer different questions. sflow exports only
# CUDA_VISIBLE_DEVICES (Backend.resource_env pops NVIDIA_VISIBLE_DEVICES -- it is
# consumed at container CREATION and setting it carved the container down before
# the plan could be applied). So whatever NVIDIA_VISIBLE_DEVICES holds here came
# from the runtime, and recording it verbatim is how a reader attributes a
# surprising device set to that layer rather than to sflow.
__sflow_in="${CUDA_VISIBLE_DEVICES:-}"
# Describe a variable VERBATIM. UNSET and SET-BUT-EMPTY are different states and
# must not read alike, so they get <env-not-set> and <set-as-empty> -- but no
# interpretation beyond that: what "unset" or "all" means to CUDA or to a
# container runtime is the reader's call, and a gloss here would be sflow
# asserting semantics it does not own (and would be wrong on any stack that
# differs). One function, so the two sentinels have one spelling.
__sflow_desc() {
  [ -n "${!1+x}" ] || { printf '<env-not-set>'; return; }
  printf '%s' "${!1:-<set-as-empty>}"
}
__sflow_in_desc="$(__sflow_desc CUDA_VISIBLE_DEVICES)"
__sflow_nvd_desc="$(__sflow_desc NVIDIA_VISIBLE_DEVICES)"

# THE RECORD FORMAT IS A CONTRACT -- three independent parsers read it:
#   * this function (producer)
#   * sflow.utils.gpu.task_gpu_record (Python; run reporting + the summary)
#   * sample_test.sh::gpu_placement_verified (shell `sed`; the e2e verdict)
# Line 1 is the bare device list and must stay parseable on its own. Every other
# line is `key=value`; `visible=`/`selected=` repeat, the rest are scalars.
# Renaming a key breaks the shell reader SILENTLY, so the key set is pinned by
# test_the_marker_format_keys_are_the_contract -- change both, or neither.
__sflow_record() {
  printf '%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'node=%s\n' "$__sflow_node"
  printf 'action=%s\n' "$__sflow_act"
  printf 'reason=%s\n' "${__sflow_reason:-(placement left to device-index arithmetic)}"
  printf 'cuda_visible_devices_inherited=%s\n' "$__sflow_in_desc"
  printf 'nvidia_visible_devices_inherited=%s\n' "$__sflow_nvd_desc"
  # No nvidia_visible_devices= counterpart: sflow never writes that variable, so
  # a post-state would equal the inherited line in every record ever produced.
  printf 'cuda_visible_devices=%s\n' "$(__sflow_desc CUDA_VISIBLE_DEVICES)"
  printf 'planned_host_indices=%s\n' "$__sflow_plan"
  printf 'planned_uuids=%s\n' "${__sflow_want:-(not resolved)}"
  printf 'visible_gpu_count=%s\n' "${#__sflow_uuid[@]}"
  for __sflow_i in "${!__sflow_uuid[@]}"; do
    printf 'visible=%s %s\n' "$__sflow_i" "${__sflow_uuid[$__sflow_i]}"
  done
  __sflow_now="${CUDA_VISIBLE_DEVICES:-}"
  for __sflow_t in ${__sflow_now//,/ }; do
    case "$__sflow_t" in
      GPU-*) printf 'selected=%s %s\n' "?" "$__sflow_t" ;;
      ''|*[!0-9]*) printf 'selected=%s (unresolvable)\n' "$__sflow_t" ;;
      *) printf 'selected=%s %s\n' "$__sflow_t" "${__sflow_uuid[$__sflow_t]:-(not visible here)}" ;;
    esac
  done
}
# Called before every exit too, not just at the end: a step that aborts on a
# mis-placement is exactly when the record is worth having.
__sflow_save() {
  [ -n "${SFLOW_TASK_OUTPUT_DIR:-}" ] && [ "${SLURM_LOCALID:-0}" = 0 ] || return 0
  __sflow_marker="${SFLOW_GPU_MARKER:-sflow_gpus.log}"
  if [ "${SLURM_STEP_NUM_NODES:-1}" = 1 ] && [ "${SLURM_PROCID:-0}" = 0 ]; then
    __sflow_record > "$SFLOW_TASK_OUTPUT_DIR/$__sflow_marker" 2>/dev/null || true
  else
    __sflow_record > "$SFLOW_TASK_OUTPUT_DIR/${__sflow_marker%.log}.$__sflow_node.log" 2>/dev/null || true
  fi
}

__sflow_done=""
__sflow_act="fallback"
__sflow_reason=""

# What this namespace can really see, probed HERE (inside the container/cgroup, so
# it is the step's own view, not the driver's). Keyed by the index nvidia-smi
# REPORTS rather than by row position, so the record states a detected index and
# an ordinal lookup stays correct even if the numbering is ever not 0..N-1.
# Indexed (NOT associative): bash indexed arrays are sparse, and ${!arr[@]}
# yields subscripts in ascending numeric order -- an associative array would
# iterate in hash order and silently scramble the ordered comparison below.
__sflow_uuid=()
while IFS=, read -r __sflow_i __sflow_u; do
  case "$__sflow_i" in ''|*[!0-9]*) continue ;; esac
  [ -n "$__sflow_u" ] && __sflow_uuid[$__sflow_i]="$__sflow_u"
done < <(timeout 10 nvidia-smi --query-gpu=index,uuid --format=csv,noheader 2>/dev/null | tr -d ' \r')
if [ -n "$__sflow_want" ]; then
  if [ "${#__sflow_uuid[@]}" -eq 0 ]; then
    # Nothing to look the planned UUIDs up against. Record that the placement is
    # unproven -- but do NOT stop here. The index arithmetic below still narrows a
    # step that was handed the whole allocation, which is the collision this
    # exists to prevent, and it is exactly what a task with NO planned UUIDs gets.
    # Returning early would make a KNOWN plan do less than an unknown one.
    __sflow_act="unverified"
    __sflow_reason="no nvidia-smi in this step, so the planned UUIDs could not be checked"
  else
    # ONE rule, whatever set the devices: look up each planned card by UUID among
    # the ones this step can actually see, and name their indices. That covers
    # every shape without special cases -- more visible than planned narrows to
    # the right subset; exactly as many still fixes a stale value that names the
    # wrong ordinals; a container renumbering from 0 is just a different lookup
    # result. UUID is the only identifier that survives every layer's renumbering.
    __sflow_sel=""
    __sflow_miss=""
    for __sflow_w in ${__sflow_want//,/ }; do
      __sflow_hit=""
      for __sflow_i in "${!__sflow_uuid[@]}"; do
        if [ "${__sflow_uuid[$__sflow_i]}" = "$__sflow_w" ]; then __sflow_hit="$__sflow_i"; break; fi
      done
      if [ -z "$__sflow_hit" ]; then __sflow_miss="$__sflow_w"; break; fi
      __sflow_sel="${__sflow_sel:+$__sflow_sel,}$__sflow_hit"
    done
    if [ -n "$__sflow_miss" ]; then
      # A planned card is not here. Who chose the devices decides what that means:
      #   * Slurm did (GRES granted them, or our CUDA_VISIBLE_DEVICES was replaced)
      #     -> it may have picked cards the planner never assumed, so the plan is a
      #     POSITION into what Slurm gave us. Fall through to index arithmetic
      #     rather than fail a healthy run.
      #   * sflow did -> the step holds something it was never given, which is the
      #     silent mis-placement this exists to catch. Fail loudly.
      #
      # Both signals below are STEP-scoped, deliberately. SLURM_JOB_GPUS is not:
      # it says the JOB has GPUs, and Backend.resource_env copies every SLURM_*
      # var out of the DRIVER's environment into every task -- so on the `batch
      # --submit` path (driver inside the sbatch job) it was set for every step on
      # any GRES cluster. Steps also run --overlap, where Slurm does not carve per
      # step and never sets SLURM_STEP_GPUS, so that leaked job-level value was the
      # only signal in play and the exit 97 below could not fire on the very
      # clusters it was written for. Do not add it back.
      if [ -n "${SLURM_STEP_GPUS:-}" ] ||
         { [ -n "$__sflow_in" ] && [ "$__sflow_in" != "$__sflow_plan" ]; }; then
        # Keep $__sflow_want: those UUIDs WERE resolved, they just are not here,
        # and that distinction is the whole diagnosis. Clearing it to signal
        # "fall through" made the record claim planned_uuids=(not resolved),
        # which points a reader at the driver probe instead of at the grant.
        # Falling through is already what an empty $__sflow_done means.
        echo "sflow: planned GPU $__sflow_miss is not among the devices Slurm granted this step; using the planned slice as a position into them instead" >&2
        __sflow_reason="planned GPU $__sflow_miss is not among the devices Slurm granted this step"
      else
        echo "sflow: planned GPU $__sflow_miss is not visible on $__sflow_node (visible: ${__sflow_uuid[*]})" >&2
        __sflow_act="missing"; __sflow_reason="planned GPU $__sflow_miss is not visible on this node"
        __sflow_save
        exit 97
      fi
    else
      # Export only when it actually differs. The record keeps both the inherited
      # and the final value, so "did this change anything?" is answered there
      # rather than by a separate action name.
      #
      # CUDA_VISIBLE_DEVICES only. NVIDIA_VISIBLE_DEVICES is consumed by the
      # container runtime at CREATION time -- the container already exists, so
      # writing it cannot change what is exposed, and the runtime has already
      # rewritten it to describe this namespace (enroot sets "all"). Overwriting
      # that with CUDA ordinals states something false about a different layer.
      [ "$__sflow_sel" = "${CUDA_VISIBLE_DEVICES:-}" ] || export CUDA_VISIBLE_DEVICES="$__sflow_sel"
      __sflow_done=1; __sflow_act="verified"
      __sflow_reason="planned GPUs located by UUID among ${#__sflow_uuid[@]} visible device(s)"
    fi
  fi
fi
if [ -z "$__sflow_done" ]; then
__sflow_seen="${CUDA_VISIBLE_DEVICES:-}"
__sflow_real=""
if [ "${#__sflow_uuid[@]}" -gt 0 ]; then
  for __sflow_i in "${!__sflow_uuid[@]}"; do
    __sflow_real="${__sflow_real:+$__sflow_real,}$__sflow_i"
  done
  for __sflow_d in ${__sflow_seen//,/ }; do
    case ",$__sflow_real," in
      *",$__sflow_d,"*) ;;
      *) __sflow_seen="" ;;
    esac
  done
  [ -n "$__sflow_seen" ] || __sflow_seen="$__sflow_real"
else
  echo "sflow: no nvidia-smi here, so CUDA_VISIBLE_DEVICES=${__sflow_seen:-<unset>} is taken on trust; a runtime that renumbered this task's devices from 0 cannot be detected, and placement may be wrong" >&2
fi
if [ -z "$__sflow_seen" ]; then
  __sflow_sel="$__sflow_plan"
else
  IFS=, read -r -a __sflow_dev <<< "$__sflow_seen"
  IFS=, read -r -a __sflow_slot <<< "$__sflow_plan"
  if [ "${#__sflow_dev[@]}" -lt "${#__sflow_slot[@]}" ]; then
    echo "sflow: step has ${#__sflow_dev[@]} GPU(s) but this task was planned for ${#__sflow_slot[@]} (CUDA_VISIBLE_DEVICES=$__sflow_seen)" >&2
    __sflow_act="too-few"; __sflow_reason="step has ${#__sflow_dev[@]} GPU(s), task was planned for ${#__sflow_slot[@]}"
    __sflow_save
    exit 97
  elif [ "${#__sflow_dev[@]}" -eq "${#__sflow_slot[@]}" ]; then
    __sflow_sel="$__sflow_seen"
  else
    __sflow_sel=""
    for __sflow_i in "${__sflow_slot[@]}"; do
      if [ -z "${__sflow_dev[$__sflow_i]:-}" ]; then
        echo "sflow: planned GPU slot $__sflow_i is outside CUDA_VISIBLE_DEVICES=$__sflow_seen" >&2
        __sflow_act="out-of-range"; __sflow_reason="planned GPU slot $__sflow_i is outside the visible devices"
        __sflow_save
        exit 97
      fi
      __sflow_sel="${__sflow_sel:+$__sflow_sel,}${__sflow_dev[$__sflow_i]}"
    done
  fi
fi
export CUDA_VISIBLE_DEVICES="$__sflow_sel"
fi
__sflow_save
