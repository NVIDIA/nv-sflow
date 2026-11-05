const assert = require("node:assert/strict");
const test = require("node:test");

const { resolveSelection } = require("./recipeSelection");

// Shaped like real recipe_data.js rows, trimmed to the fields resolution
// reads. DeepSeek-V4-Pro is the interesting one: 960 GB of weights, so its
// single-node strategies are unrunnable on h100 (640) and gb200 (768) but fine
// on b200 (1440) - the case that produced both bugs the module exists to stop.
function recipe(over) {
  return {
    model_group: "deepseek-ai/DeepSeek-V4-Pro",
    default_model: "deepseek-ai/DeepSeek-V4-Pro-0813",
    variant: "default",
    hardware: "h100",
    display_name: "H100",
    strategy: "single_node_tp",
    strategy_kind: "single_node",
    nodes: 1,
    machine: "h100.yaml",
    vram_min_gb: 960,
    node_vram_gb: 640,
    ...over,
  };
}

const MULTI = { strategy: "multi_node_tep", strategy_kind: "multi_node", nodes: 2, machine: "h100_multi.yaml" };
const B200 = { hardware: "b200", display_name: "B200", machine: "b200.yaml", node_vram_gb: 1440 };

const DATA = [
  recipe({}),
  recipe({ strategy: "single_node_dep" }),
  recipe(MULTI),
  recipe({ ...MULTI, strategy: "multi_node_dep" }),
  recipe({ ...B200 }),
  recipe({ ...B200, strategy: "single_node_dep" }),
  recipe({ ...B200, variant: "fp8" }),
  recipe({ variant: "fp8", ...MULTI }),
  recipe({ model_group: "Qwen/Qwen3.5-0.8B", default_model: "Qwen/Qwen3.5-0.8B", vram_min_gb: 2, node_vram_gb: 640 }),
];

// --- the two bugs this module was extracted to prevent -----------------------

test("switching launcher keeps the selected model", () => {
  const wanted = { launcher: "slurm", modelGroup: "deepseek-ai/DeepSeek-V4-Pro", hardware: "b200" };
  const before = resolveSelection(DATA, wanted);
  const after = resolveSelection(DATA, { ...wanted, launcher: "docker" });

  assert.equal(before.selected.modelGroup, "deepseek-ai/DeepSeek-V4-Pro");
  assert.equal(after.selected.modelGroup, "deepseek-ai/DeepSeek-V4-Pro");
});

test("no recipe is resolved when every strategy on offer is disabled", () => {
  // docker reaches single-node only, and 960 GB fits on no h100 node.
  const r = resolveSelection(DATA, {
    launcher: "docker",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    hardware: "h100",
  });

  assert.equal(r.recipe, null);
  assert.equal(r.selected.strategy, null);
  assert.deepEqual(r.blocked, { kind: "vram", needGb: 960, haveGb: 640 });
  assert.deepEqual(r.machineFiles, []);
  assert.ok(r.axes.strategies.every((s) => s.disabled));
});

// --- the axis-settling rule, once per axis -----------------------------------

test("an unreachable model falls back to the first one offered", () => {
  // Offered in display order (the sidebar's), so the fallback is the first
  // model the user can actually see, not the first row in the data file.
  const r = resolveSelection(DATA, { launcher: "slurm", modelGroup: "does-not/exist" });
  assert.equal(r.selected.modelGroup, "deepseek-ai/DeepSeek-V4-Pro");
  assert.deepEqual(r.axes.models.map((m) => m.group), [
    "deepseek-ai/DeepSeek-V4-Pro",
    "Qwen/Qwen3.5-0.8B",
  ]);
});

test("a variant the model lacks falls back", () => {
  const r = resolveSelection(DATA, {
    launcher: "slurm",
    modelGroup: "Qwen/Qwen3.5-0.8B",
    variant: "nvfp4",
  });
  assert.equal(r.selected.variant, "default");
});

test("the checkpoint outranks the hardware when the two disagree", () => {
  // The axes settle in the order the page stacks them - checkpoint above
  // hardware - so asking for a variant that exists only on other hardware
  // keeps the variant and moves the hardware, not the reverse.
  const r = resolveSelection(DATA, {
    launcher: "docker",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    variant: "fp8",
    hardware: "h100",
  });
  assert.equal(r.selected.variant, "fp8");
  assert.equal(r.selected.hardware, "b200");
});

test("a variant still on offer survives a hardware change", () => {
  const r = resolveSelection(DATA, {
    launcher: "slurm",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    variant: "fp8",
    hardware: "b200",
  });
  assert.equal(r.selected.variant, "fp8");
});

test("hardware the variant is not built for falls back", () => {
  const r = resolveSelection(DATA, {
    launcher: "slurm",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    hardware: "gb200",
  });
  assert.ok(["h100", "b200"].includes(r.selected.hardware));
});

test("a disabled strategy is never settled on while a runnable one exists", () => {
  const r = resolveSelection(DATA, {
    launcher: "slurm",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    hardware: "h100",
    strategy: "single_node_tp",
  });

  assert.notEqual(r.selected.strategy, "single_node_tp");
  assert.equal(r.recipe.strategy_kind, "multi_node");
  assert.equal(r.blocked, null);
});

test("a runnable strategy the user picked is kept", () => {
  const r = resolveSelection(DATA, {
    launcher: "slurm",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    hardware: "h100",
    strategy: "multi_node_dep",
  });
  assert.equal(r.selected.strategy, "multi_node_dep");
});

// --- what each launcher may reach --------------------------------------------

test("docker and k8s hide multi-node recipes, slurm shows them", () => {
  const pick = { modelGroup: "deepseek-ai/DeepSeek-V4-Pro", hardware: "h100" };
  const kinds = (launcher) =>
    resolveSelection(DATA, { ...pick, launcher }).axes.strategies.map((s) => s.value);

  assert.ok(kinds("slurm").includes("multi_node_tep"));
  assert.ok(!kinds("docker").some((s) => s.startsWith("multi_node")));
  assert.ok(!kinds("k8s").some((s) => s.startsWith("multi_node")));
});

// --- which machine fragments the command composes ----------------------------

test("slurm composes the gpu fragment with the single-node backend", () => {
  const r = resolveSelection(DATA, {
    launcher: "slurm",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    hardware: "b200",
    strategy: "single_node_tp",
  });
  assert.deepEqual(r.machineFiles, ["gpu/b200.yaml", "backend/slurm.yaml"]);
});

test("a multi-node recipe takes the multi-node slurm backend, on the same gpu fragment", () => {
  const r = resolveSelection(DATA, {
    launcher: "slurm",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    hardware: "h100",
    strategy: "multi_node_tep",
  });
  assert.deepEqual(r.machineFiles, ["gpu/h100.yaml", "backend/slurm_multi.yaml"]);
});

test("naming a PVC adds the k8s model-pvc fragment, leaving it out keeps two", () => {
  const pick = {
    launcher: "k8s",
    modelGroup: "deepseek-ai/DeepSeek-V4-Pro",
    hardware: "b200",
    strategy: "single_node_tp",
  };

  assert.deepEqual(resolveSelection(DATA, pick).machineFiles, [
    "gpu/b200.yaml",
    "backend/k8s.yaml",
  ]);
  assert.deepEqual(resolveSelection(DATA, { ...pick, pvcClaim: "models-pvc" }).machineFiles, [
    "gpu/b200.yaml",
    "backend/k8s.yaml",
    "backend/k8s_model_pvc.yaml",
  ]);
  assert.deepEqual(
    resolveSelection(DATA, { ...pick, launcher: "docker", pvcClaim: "models-pvc" }).machineFiles,
    ["gpu/b200.yaml", "backend/docker.yaml"],
  );
});

// --- the search box narrows the model list -----------------------------------

test("the filter narrows the models on offer and can move the selection", () => {
  const r = resolveSelection(DATA, { launcher: "slurm", filter: "qwen" });
  assert.deepEqual(r.axes.models.map((m) => m.group), ["Qwen/Qwen3.5-0.8B"]);
  assert.equal(r.selected.modelGroup, "Qwen/Qwen3.5-0.8B");
});

// --- recipes indexed without size info are treated as fitting ----------------

test("a recipe with no vram numbers is never disabled", () => {
  const data = [recipe({ vram_min_gb: undefined, node_vram_gb: undefined })];
  const r = resolveSelection(data, { launcher: "docker" });

  assert.equal(r.axes.strategies[0].disabled, null);
  assert.ok(r.recipe);
});
