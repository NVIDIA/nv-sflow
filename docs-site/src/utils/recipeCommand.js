// Turns a resolved recipe selection plus the user's inputs into the
// `sflow run` command - as data first, text second.
//
// Three things need the same command with different paths: the page shows it
// with paths relative to a vllm-recipes-sflow checkout, the downloadable
// run.sh needs the same files flat beside it, and the bundle needs the file
// list itself. The page used to build the display string, and the bundle then
// read that string back out of the DOM and String.replace()d each path - so a
// formatting tweak to the display could quietly break the download. Now the
// plan is the source of truth and each consumer renders it.
//
// SYNTAX CONSTRAINT - CommonJS bundled by Docusaurus's webpack: no for...of,
// no object/array spread, no classes, no async. Docusaurus's babel preset
// injects ESM runtime helpers for those, which flips the file to an ES module
// and makes module.exports throw at page load. See recipeSelection.js.

// The --set flags each launcher's fragments expect, in the order the command
// lists them. `always` ones are emitted even when the user left them blank -
// the caller supplies the placeholder text so the checklist can flag it.
// The rest appear only when filled in: their fragment has a usable default
// or, for the Slurm pair, editing the fragment is the other valid place.
function launcherSets(launcher, inputs) {
  if (launcher === "docker") {
    return [["MODEL_HOST_PATH", inputs.modelHostPath]];
  }
  if (launcher === "k8s") {
    const sets = [["K8S_NAMESPACE", inputs.k8sNamespace]];
    if (inputs.pvcClaim) sets.push(["MODEL_PVC_CLAIM", inputs.pvcClaim]);
    return sets;
  }
  const sets = [];
  if (inputs.slurmAccount) sets.push(["SLURM_ACCOUNT", inputs.slurmAccount]);
  if (inputs.slurmPartition) sets.push(["SLURM_PARTITION", inputs.slurmPartition]);
  return sets;
}

// Where machines/backend/k8s_model_pvc.yaml mounts the PVC (its
// MODEL_MOUNT_PATH default). The only path a k8s task pod keeps.
const K8S_MODEL_MOUNT = "/models";

function isUnder(path, root) {
  const r = root.replace(/\/+$/, "");
  return path === r || path.indexOf(r + "/") === 0;
}

// Things about this plan that will bite at run time but leave the command
// itself well-formed. Structured, not prose: the page words them.
//
//   no_docker_image      recipe generated before vllm-recipes-sflow moved the
//                        image into a DOCKER_IMAGE variable; it still sets
//                        container_image per task, an srun-only key that
//                        docker_run/k8s reject outright.
//   cache_unmounted      CACHE_DIR set on k8s with no PVC: the pod mounts no
//                        volume, so nothing written there survives.
//   cache_outside_mount  CACHE_DIR set but outside the one path the container
//                        keeps, so the checkpoint downloads again every run -
//                        silently, since the run itself succeeds (a
//                        DeepSeek-V4-Flash run threw away 167 GB that way).
//   cache_read_only      the selected PVC is mounted read-only, so creating
//                        caches beneath it would fail.
function warningsFor(inputs) {
  const warnings = [];
  const containerLauncher = inputs.launcher !== "slurm";

  if (containerLauncher && inputs.recipeYaml !== undefined &&
      inputs.recipeYaml.indexOf("\n  DOCKER_IMAGE:") < 0) {
    warnings.push({ kind: "no_docker_image" });
  }

  if (containerLauncher && inputs.cacheDir != null) {
    const mount = inputs.launcher === "docker"
      ? inputs.modelHostPath
      : (inputs.pvcClaim ? K8S_MODEL_MOUNT : "");
    if (!mount) {
      warnings.push({ kind: "cache_unmounted", cacheDir: inputs.cacheDir });
    } else if (!isUnder(inputs.cacheDir, mount)) {
      warnings.push({
        kind: "cache_outside_mount",
        cacheDir: inputs.cacheDir,
        mount: mount,
        mountName: inputs.launcher === "docker" ? "MODEL_HOST_PATH" : "the model PVC mount",
      });
    } else if (inputs.launcher === "k8s" && inputs.pvcReadOnly) {
      warnings.push({ kind: "cache_read_only", cacheDir: inputs.cacheDir });
    }
  }
  return warnings;
}

// inputs:
//   launcher        "slurm" | "docker" | "k8s"
//   modelHostPath   docker: host dir bind-mounted into every container (always emitted)
//   k8sNamespace    k8s: namespace for the task pods (always emitted)
//   pvcClaim        k8s: existing PVC holding the weights, or ""
//   pvcReadOnly     whether the selected PVC fragment mounts it read-only
//   slurmAccount, slurmPartition   slurm: "" leaves them to the fragment
//   localModel      path of a local checkpoint, or null to download from HF
//   ifaceName       rendezvous NIC for multi-node recipes, or ""
//   cacheDir        shared cache root, or null to leave caches in place
//   mode            "--tui" | "--dry-run" | "" (submit, don't attach)
//   recipeYaml      the recipe's workflow.yaml text, or undefined while it is
//                   still loading (skips the checks that need it)
function planCommand(resolution, inputs) {
  const recipe = resolution.recipe;
  if (!recipe) return null;
  // A remembered option must stop affecting the plan when the user switches
  // to a recipe that does not consume CACHE_DIR. Missing legacy metadata keeps
  // its previous behavior; current indexed recipes always declare this flag.
  if (recipe.has_cache_dir === false) {
    inputs = Object.assign({}, inputs, { cacheDir: null });
  }

  const files = [recipe.path].concat(
    resolution.machineFiles.map(function (f) { return "machines/" + f; }),
  );

  const sets = launcherSets(inputs.launcher, inputs);
  if (inputs.localModel != null) sets.push(["MODEL", inputs.localModel]);
  // Only the multi-node/pd_cluster recipes declare IFACE_NAME (recipe_data's
  // has_iface); a --set for a variable the recipe does not have is an error.
  if (recipe.has_iface && inputs.ifaceName) sets.push(["IFACE_NAME", inputs.ifaceName]);
  if (inputs.cacheDir != null) sets.push(["CACHE_DIR", inputs.cacheDir]);

  return { files: files, sets: sets, mode: inputs.mode || "", warnings: warningsFor(inputs) };
}

function basename(path) {
  return path.split("/").pop();
}

function shellQuote(value) {
  const text = String(value);
  if (/^[A-Za-z0-9_@%+=:,./-]+$/.test(text)) return text;
  return "'" + text.replace(/'/g, "'\"'\"'") + "'";
}

// One argument per line, continued with backslashes, the way the page has
// always shown it. `flat` renders each file by basename - for run.sh, which
// sits in one directory with the files beside it.
function renderCommand(plan, options) {
  const flat = !!(options && options.flat);
  const args = plan.files
    .map(function (f) { return "-f " + shellQuote(flat ? basename(f) : f); })
    .concat(plan.sets.map(function (kv) { return "--set " + shellQuote(kv[0] + "=" + kv[1]); }));
  if (plan.mode) args.push(plan.mode);
  const lines = ["sflow run"].concat(args.map(function (a) { return "  " + a; }));
  return lines.join(" \\\n");
}

module.exports = { planCommand, renderCommand, basename };
