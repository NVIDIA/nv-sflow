const assert = require("node:assert/strict");
const test = require("node:test");

const { planCommand, renderCommand } = require("./recipeCommand");

// A resolveSelection() result trimmed to what planCommand reads.
function resolution(over) {
  return Object.assign(
    {
      recipe: {
        path: "recipes/Qwen/Qwen3.5-0.8B/workflow.yaml",
        has_iface: false,
      },
      machineFiles: ["gpu/gb200.yaml", "backend/slurm.yaml"],
    },
    over || {},
  );
}

const BLANK = {
  launcher: "slurm",
  modelHostPath: "",
  k8sNamespace: "",
  pvcClaim: "",
  slurmAccount: "",
  slurmPartition: "",
  localModel: null,
  ifaceName: "",
  cacheDir: null,
  mode: "--tui",
};

function inputs(over) {
  return Object.assign({}, BLANK, over || {});
}

// --- the plan ----------------------------------------------------------------

test("files are the recipe then every machine fragment, repo-relative", () => {
  const plan = planCommand(resolution(), inputs());
  assert.deepEqual(plan.files, [
    "recipes/Qwen/Qwen3.5-0.8B/workflow.yaml",
    "machines/gpu/gb200.yaml",
    "machines/backend/slurm.yaml",
  ]);
});

test("no recipe, no plan", () => {
  assert.equal(planCommand(resolution({ recipe: null, machineFiles: [] }), inputs()), null);
});

test("slurm emits account/partition only when filled in", () => {
  assert.deepEqual(planCommand(resolution(), inputs()).sets, []);
  assert.deepEqual(
    planCommand(resolution(), inputs({ slurmAccount: "perflab", slurmPartition: "gb200" })).sets,
    [["SLURM_ACCOUNT", "perflab"], ["SLURM_PARTITION", "gb200"]],
  );
});

test("docker always emits MODEL_HOST_PATH, with whatever the caller supplied", () => {
  const plan = planCommand(
    resolution({ machineFiles: ["gpu/b200.yaml", "backend/docker.yaml"] }),
    inputs({ launcher: "docker", modelHostPath: "/path/to/model-weights/on-this-box" }),
  );
  assert.deepEqual(plan.sets, [["MODEL_HOST_PATH", "/path/to/model-weights/on-this-box"]]);
});

test("k8s always emits the namespace and adds the PVC only when named", () => {
  const k8s = resolution({ machineFiles: ["gpu/gb200.yaml", "backend/k8s.yaml"] });
  assert.deepEqual(
    planCommand(k8s, inputs({ launcher: "k8s", k8sNamespace: "nim-craft" })).sets,
    [["K8S_NAMESPACE", "nim-craft"]],
  );
  assert.deepEqual(
    planCommand(k8s, inputs({ launcher: "k8s", k8sNamespace: "nim-craft", pvcClaim: "models-pvc" })).sets,
    [["K8S_NAMESPACE", "nim-craft"], ["MODEL_PVC_CLAIM", "models-pvc"]],
  );
});

test("local checkpoint, cache dir and mode ride along after the launcher's sets", () => {
  const plan = planCommand(
    resolution(),
    inputs({ slurmAccount: "a", localModel: "/ckpt/qwen", cacheDir: "/lustre/cache", mode: "--dry-run" }),
  );
  assert.deepEqual(plan.sets, [
    ["SLURM_ACCOUNT", "a"],
    ["MODEL", "/ckpt/qwen"],
    ["CACHE_DIR", "/lustre/cache"],
  ]);
  assert.equal(plan.mode, "--dry-run");
});

test("IFACE_NAME is emitted only for recipes that declare it", () => {
  const withIface = resolution({ recipe: { path: "recipes/x/workflow.yaml", has_iface: true } });
  assert.deepEqual(planCommand(withIface, inputs({ ifaceName: "ib0" })).sets, [["IFACE_NAME", "ib0"]]);
  assert.deepEqual(planCommand(resolution(), inputs({ ifaceName: "ib0" })).sets, []);
  assert.deepEqual(planCommand(withIface, inputs()).sets, []);
});

// --- warnings: well-formed commands that would still bite ---------------------

const OLD_YAML = "variables:\n  MODEL:\n    value: x\nworkflow:\n  tasks:\n    - operator:\n        container_image: y\n";
const NEW_YAML = "variables:\n  MODEL:\n    value: x\n  DOCKER_IMAGE:\n    value: y\n";

test("a recipe without DOCKER_IMAGE is flagged on docker/k8s, never on slurm, never while loading", () => {
  const kinds = (over) => planCommand(resolution(), inputs(over)).warnings.map((w) => w.kind);
  assert.deepEqual(kinds({ launcher: "docker", modelHostPath: "/m", recipeYaml: OLD_YAML }), ["no_docker_image"]);
  assert.deepEqual(kinds({ launcher: "k8s", k8sNamespace: "ns", recipeYaml: OLD_YAML }), ["no_docker_image"]);
  assert.deepEqual(kinds({ launcher: "slurm", recipeYaml: OLD_YAML }), []);
  assert.deepEqual(kinds({ launcher: "docker", modelHostPath: "/m", recipeYaml: NEW_YAML }), []);
  assert.deepEqual(kinds({ launcher: "docker", modelHostPath: "/m", recipeYaml: undefined }), []);
});

test("docker: CACHE_DIR must sit under MODEL_HOST_PATH", () => {
  const plan = (cacheDir) =>
    planCommand(resolution(), inputs({ launcher: "docker", modelHostPath: "/raid/cache", cacheDir, recipeYaml: NEW_YAML }));
  assert.deepEqual(plan("/raid/cache").warnings, []);
  assert.deepEqual(plan("/raid/cache/hf").warnings, []);
  assert.deepEqual(plan("/home/me/cache").warnings, [
    { kind: "cache_outside_mount", cacheDir: "/home/me/cache", mount: "/raid/cache", mountName: "MODEL_HOST_PATH" },
  ]);
  // a sibling that merely shares the prefix is still outside
  assert.equal(plan("/raid/cache2").warnings[0].kind, "cache_outside_mount");
});

test("k8s: CACHE_DIR has nowhere to go without a PVC, and must sit under /models with one", () => {
  const plan = (over) =>
    planCommand(resolution(), inputs(Object.assign({ launcher: "k8s", k8sNamespace: "ns", recipeYaml: NEW_YAML }, over)));
  assert.deepEqual(plan({ cacheDir: "/scratch" }).warnings, [{ kind: "cache_unmounted", cacheDir: "/scratch" }]);
  assert.deepEqual(plan({ cacheDir: "/models/hf", pvcClaim: "pvc" }).warnings, []);
  assert.deepEqual(plan({ cacheDir: "/scratch", pvcClaim: "pvc" }).warnings, [
    { kind: "cache_outside_mount", cacheDir: "/scratch", mount: "/models", mountName: "the model PVC mount" },
  ]);
});

test("slurm writes to the host filesystem, so CACHE_DIR is never a warning there", () => {
  const plan = planCommand(resolution(), inputs({ launcher: "slurm", cacheDir: "/anywhere", recipeYaml: OLD_YAML }));
  assert.deepEqual(plan.warnings, []);
  assert.deepEqual(planCommand(resolution(), inputs({ launcher: "docker", modelHostPath: "/m", cacheDir: null, recipeYaml: NEW_YAML })).warnings, []);
});

// --- rendering ---------------------------------------------------------------

test("renders one argument per line with continuations and the mode last", () => {
  const plan = planCommand(resolution(), inputs({ slurmAccount: "perflab", mode: "--tui" }));
  assert.equal(
    renderCommand(plan),
    [
      "sflow run \\",
      "  -f recipes/Qwen/Qwen3.5-0.8B/workflow.yaml \\",
      "  -f machines/gpu/gb200.yaml \\",
      "  -f machines/backend/slurm.yaml \\",
      "  --set SLURM_ACCOUNT=perflab \\",
      "  --tui",
    ].join("\n"),
  );
});

test("an empty mode leaves no dangling continuation", () => {
  const text = renderCommand(planCommand(resolution(), inputs({ mode: "" })));
  assert.ok(text.endsWith("  -f machines/backend/slurm.yaml"));
  assert.ok(!text.endsWith("\\"));
});

test("flat rendering keeps only basenames, for run.sh beside its files", () => {
  const plan = planCommand(
    resolution({ machineFiles: ["gpu/gb200.yaml", "backend/k8s.yaml", "backend/k8s_model_pvc.yaml"] }),
    inputs({ launcher: "k8s", k8sNamespace: "ns", pvcClaim: "pvc", mode: "" }),
  );
  assert.equal(
    renderCommand(plan, { flat: true }),
    [
      "sflow run \\",
      "  -f workflow.yaml \\",
      "  -f gb200.yaml \\",
      "  -f k8s.yaml \\",
      "  -f k8s_model_pvc.yaml \\",
      "  --set K8S_NAMESPACE=ns \\",
      "  --set MODEL_PVC_CLAIM=pvc",
    ].join("\n"),
  );
});
