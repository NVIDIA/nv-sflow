const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const vm = require("node:vm");
const { gunzipSync } = require("node:zlib");
const { execFileSync } = require("node:child_process");
const yaml = require("js-yaml");
const { planCommand, renderCommand } = require("./recipeCommand");
const { resolveSelection } = require("./recipeSelection");
const { composeYaml } = require("./recipeCompose");

const data = vm.runInNewContext(
  fs.readFileSync(require.resolve("../../static/vllm-recipes/recipe_data.js"), "utf8") +
  ";({recipes: RECIPE_DATA, machines: MACHINE_DATA})",
);
const recipeYaml = JSON.parse(gunzipSync(fs.readFileSync(
  require.resolve("../../static/vllm-recipes/recipe_yaml.json.gz"),
)));

function selection(modelGroup, launcher = "k8s") {
  return resolveSelection(data.recipes, {
    modelGroup, launcher, hardware: "gb200", pvcClaim: "models-pvc",
  });
}

test("rendered shell arguments preserve literal paths in both command formats", {
  skip: process.platform === "win32",
}, () => {
  const values = ["/models/local model", "/models/O'Brien", "/models/$USER/$(echo expanded);&*", "/models/a\nb"];
  for (const flat of [false, true]) {
    for (const value of values) {
      const plan = { files: ["recipes/a b/workflow.yaml"], sets: [["MODEL", value]], mode: "--dry-run" };
      const command = renderCommand(plan, { flat });
      // A local shell function captures argv; no sflow or workload is executed.
      const args = execFileSync("/bin/sh", ["-c", 'sflow() { printf "%s\\0" "$@"; };\n' + command])
        .toString().split("\0").slice(0, -1);
      assert.deepEqual(args, ["run", "-f", flat ? "workflow.yaml" : plan.files[0], "--set", "MODEL=" + value, "--dry-run"]);
    }
  }
});

test("unsupported recipe omits remembered cache input and cache warnings, supported recipe restores it", () => {
  const inputs = { launcher: "k8s", k8sNamespace: "ns", pvcClaim: "models-pvc", pvcReadOnly: true, cacheDir: "/models/cache" };
  const supported = selection("deepseek-ai/DeepSeek-V4-Flash");
  const unsupported = selection("Qwen/Qwen3.5-0.8B");
  assert.equal(unsupported.recipe.has_cache_dir, false);
  assert.equal(supported.recipe.has_cache_dir, true);
  assert.ok(planCommand(supported, inputs).sets.some(([key]) => key === "CACHE_DIR"));
  const off = planCommand(unsupported, inputs);
  assert.ok(!off.sets.some(([key]) => key === "CACHE_DIR"));
  assert.deepEqual(off.warnings, []);
  assert.ok(planCommand(supported, inputs).sets.some(([key]) => key === "CACHE_DIR"));
});

test("cache inside a read-only PVC is warned, while writable or unused PVCs are not", () => {
  const resolved = selection("deepseek-ai/DeepSeek-V4-Flash");
  const volume = yaml.load(data.machines["backend/k8s_model_pvc.yaml"]).backends[0].volumes[0];
  const inputs = { launcher: "k8s", k8sNamespace: "ns", pvcClaim: "models-pvc", cacheDir: "/models/cache", pvcReadOnly: volume.read_only !== false };
  assert.deepEqual(planCommand(resolved, inputs).warnings, [{kind: "cache_read_only", cacheDir: "/models/cache"}]);
  assert.deepEqual(planCommand(resolved, {...inputs, pvcReadOnly: false}).warnings, []);
  assert.deepEqual(planCommand(resolved, {...inputs, cacheDir: null}).warnings, []);
});

test("single-file recipe preserves the plan's variable values across all launchers", () => {
  for (const launcher of ["slurm", "docker", "k8s"]) {
    const resolved = selection("deepseek-ai/DeepSeek-V4-Flash", launcher);
    const inputs = {
      launcher, slurmAccount: "my-account", slurmPartition: "gpu", modelHostPath: "/models",
      k8sNamespace: "ns", pvcClaim: "models-pvc", localModel: "/models/O'Brien = checkpoint",
      cacheDir: "/models/cache", mode: "--dry-run",
    };
    const plan = planCommand(resolved, inputs);
    const fragments = [{label: resolved.recipe.path, text: recipeYaml[resolved.recipe.path]}].concat(
      resolved.machineFiles.map(f => ({label: f, text: data.machines[f]})),
    );
    const composed = yaml.load(composeYaml(fragments, plan.sets));
    for (const [key, value] of plan.sets) {
      assert.equal(composed.variables.find(v => v.name === key).value, value, `${launcher}: ${key}`);
    }
    assert.ok(composed.variables.find(v => v.name === "CACHE_DIR").description);
  }
});

test("single-file override rejects an unknown variable and preserves YAML-like text", () => {
  const fragments = [{label: "wf", text: "variables:\n  MODEL:\n    value: old\n    description: Checkpoint\n"}];
  assert.throws(() => composeYaml(fragments, [["UNKNOWN", "value"]]), /UNKNOWN.*not defined/);
  for (const value of ["false", "123", "a: b", "a\nb", "[one, two]"]) {
    const out = yaml.load(composeYaml(fragments, [["MODEL", value]]));
    assert.deepEqual(out.variables[0], {name: "MODEL", value, description: "Checkpoint"});
  }
});
