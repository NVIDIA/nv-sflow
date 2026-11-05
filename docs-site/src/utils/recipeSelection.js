// Resolves what the vLLM Recipes command builder is currently pointing at.
//
// The page has five axes - launcher, model, checkpoint variant, hardware,
// strategy - and they constrain each other: docker and k8s only reach
// single-node recipes, a model exists on some hardware and not others, and a
// single-node strategy is unrunnable when the checkpoint is larger than one
// node's VRAM. Whoever answers "given what the user clicked, what is actually
// selected, what may they click next, and what is greyed out" owns the whole
// rule set.
//
// That answer used to be computed in four render functions, each re-picking
// one axis inside the DOM code that drew it (builder.js renderOptions /
// renderVariants / renderHardware / renderStrategy), with the five selections
// as module-level mutable state. Two bugs shipped from that arrangement in one
// day: clicking a launcher cleared every selection and jumped back to the
// first model, and the strategy re-pick fell back to a strategy the page
// itself had disabled, producing a runnable-looking command for a
// configuration that can only OOM. Both were one copy of the rule disagreeing
// with the other three.
//
// So it lives here instead: one pure function over plain data, no DOM, with
// the render layer reduced to painting what it returns.
//
// SYNTAX CONSTRAINT - this file is CommonJS (module.exports) so node --test can
// require it, and it is also bundled by Docusaurus's webpack. Docusaurus's own
// babel preset runs @babel/plugin-transform-runtime with useESModules: true,
// so any syntax babel lowers through a helper (for...of, object spread
// {...o}, destructuring in some forms, classes, async) gets an ESM
// `import ... from "@babel/runtime/helpers/esm/..."` injected at the top -
// which makes webpack treat the whole file as an ES module and throw
// "ES Modules may not assign module.exports" at runtime. The sibling utils
// (frameEscape.js, hashScroll.js) only avoid this because they happen to use
// none of that syntax. Stick to forEach / indexed loops / Object.assign here,
// or the page dies at load with no build-time warning.

// Docker runs one box; the k8s fragment is one pod per task (plain `k8s`
// operator, no MPI launcher). Neither does cross-node orchestration, so only
// single-node recipes are reachable through them - multi-node and pd_cluster
// stay Slurm-only.
const LAUNCHERS = [
  {
    id: "slurm",
    label: "Slurm",
    title: "sbatch/srun on a cluster — every recipe, including multi-node and prefill/decode",
  },
  {
    id: "docker",
    label: "Docker",
    title: "one box with the NVIDIA Container Toolkit — single-node recipes only",
  },
  {
    id: "k8s",
    label: "Kubernetes",
    title: "one pod per task — single-node recipes only",
  },
];

// taxonomy.yaml's "gb200" hardware id names the 4-GPU compute tray
// (display_name "GB200 NVL4") - the unit a cluster launch actually targets is
// the NVL72 rack those trays are joined into.
const HARDWARE_LABELS = { gb200: "GB200 NVL72" };

function hardwareLabel(recipe) {
  return HARDWARE_LABELS[recipe.hardware] || recipe.display_name || recipe.hardware;
}

// Upstream's variant *key* for the recommended checkpoint is literally
// "default" - not a name like the others (fp8/nvfp4/dspark). Where the served
// checkpoint id differs from the bare model name (deepseek-ai/
// DeepSeek-V4-Pro-0813), show that difference instead: it is the specific
// thing "default" stands in for.
function variantLabel(recipe) {
  if (recipe.variant !== "default") return recipe.variant;
  const base = recipe.model_group.split("/").pop();
  const served = (recipe.default_model || "").split("/").pop();
  if (served && served !== base && served.startsWith(base)) {
    const suffix = served.slice(base.length).replace(/^[-_]/, "");
    if (suffix) return suffix;
  }
  return "default";
}

// A single-node strategy shards the weights across one node's GPUs and can
// grow no further, so it is unrunnable when the checkpoint is larger than that
// node's VRAM. Same rule as fitsSingleNode() in the upstream site's
// src/lib/command-synthesis.js, on the same two numbers (see
// vllm-recipes-sflow tools/build_recipe_index.py --recipes-repo). A recipe
// indexed without those numbers counts as fitting, as upstream also treats
// missing size info as a fit.
function vramBlock(recipe) {
  if (recipe.strategy_kind !== "single_node") return null;
  if (!recipe.vram_min_gb || !recipe.node_vram_gb) return null;
  if (recipe.vram_min_gb <= recipe.node_vram_gb) return null;
  return { kind: "vram", needGb: recipe.vram_min_gb, haveGb: recipe.node_vram_gb };
}

function hardwareFileFor(recipe) {
  return (recipe.machine || "gb200.yaml").replace(/_multi\.yaml$/, ".yaml");
}

// Every launcher composes the same two halves: hardware facts, then the
// fragment binding them to a backend and to the recipe's
// `operator: {name: vllm_runtime}`. Paths are relative to machines/, in the
// order the command passes them. k8s needs nothing to pre-exist in the
// cluster - the checkpoint downloads into the pod; naming a PVC adds the
// fragment that mounts it so the download persists.
function machineFilesFor(recipe, launcher, pvcClaim) {
  const backend =
    launcher === "slurm" && recipe.strategy_kind !== "single_node" ? "slurm_multi" : launcher;
  const files = [`gpu/${hardwareFileFor(recipe)}`, `backend/${backend}.yaml`];
  if (launcher === "k8s" && pvcClaim) files.push("backend/k8s_model_pvc.yaml");
  return files;
}

// Keep what the user asked for when it is still on offer, otherwise take the
// first option. Upstream axes narrow the downstream ones, so a downstream pick
// survives a change above it whenever it remains reachable - clearing it (as
// the render functions used to) threw away a choice the user had made and
// could still have.
function settle(wantedValue, options, valueOf) {
  const kept = options.find((o) => valueOf(o) === wantedValue);
  if (kept) return wantedValue;
  return options.length ? valueOf(options[0]) : null;
}

function resolveSelection(recipes, wanted = {}) {
  const launcher = settle(wanted.launcher, LAUNCHERS, (l) => l.id) || LAUNCHERS[0].id;

  const reachable =
    launcher === "slurm" ? recipes : recipes.filter((r) => r.strategy_kind === "single_node");

  const query = (wanted.filter || "").trim().toLowerCase();
  const groups = [];
  const seen = new Set();
  reachable.forEach((r) => {
    if (seen.has(r.model_group)) return;
    if (query && !r.model_group.toLowerCase().includes(query)) return;
    seen.add(r.model_group);
    groups.push({
      group: r.model_group,
      org: r.model_group.split("/")[0],
      label: r.model_group.split("/").slice(1).join("/"),
    });
  });
  groups.sort((a, b) => a.group.localeCompare(b.group));
  const modelGroup = settle(wanted.modelGroup, groups, (g) => g.group);

  const ofModel = reachable.filter((r) => r.model_group === modelGroup);
  const variants = [];
  ofModel.forEach((r) => {
    if (variants.some((v) => v.value === r.variant)) return;
    variants.push({ value: r.variant, label: variantLabel(r) });
  });
  variants.sort((a, b) => a.value.localeCompare(b.value));
  const variant = settle(wanted.variant, variants, (v) => v.value);

  const ofVariant = ofModel.filter((r) => r.variant === variant);
  const hardware = [];
  ofVariant.forEach((r) => {
    if (hardware.some((h) => h.id === r.hardware)) return;
    hardware.push({ id: r.hardware, label: hardwareLabel(r) });
  });
  const hardwareId = settle(wanted.hardware, hardware, (h) => h.id);

  const ofHardware = ofVariant.filter((r) => r.hardware === hardwareId);
  const strategies = ofHardware.map((r) => ({
    value: r.strategy,
    nodes: r.nodes,
    disabled: vramBlock(r),
  }));
  // Never settle on a disabled strategy: under docker/k8s the list is filtered
  // to single-node, so a model needing more than one node has every remaining
  // option disabled, and falling back to the first would render a command the
  // page itself marks impossible.
  const runnable = strategies.filter((s) => !s.disabled);
  const strategy = settle(wanted.strategy, runnable, (s) => s.value);

  const recipe = strategy
    ? ofHardware.find((r) => r.strategy === strategy) || null
    : null;
  // Everything on offer here is out of reach - report why using the first
  // entry, they all fail on the same node size.
  const blocked = !runnable.length && strategies.length ? strategies[0].disabled : null;

  const mark = (list, valueOf, chosen) =>
    list.map((o) => Object.assign({}, o, { selected: valueOf(o) === chosen }));

  return {
    recipe,
    machineFiles: recipe ? machineFilesFor(recipe, launcher, wanted.pvcClaim) : [],
    selected: { launcher, modelGroup, variant, hardware: hardwareId, strategy },
    axes: {
      launchers: mark(LAUNCHERS, (l) => l.id, launcher),
      models: mark(groups, (g) => g.group, modelGroup),
      variants: mark(variants, (v) => v.value, variant),
      hardware: mark(hardware, (h) => h.id, hardwareId),
      strategies: mark(strategies, (s) => s.value, strategy),
    },
    blocked,
  };
}

module.exports = { resolveSelection, hardwareLabel, variantLabel, LAUNCHERS };
