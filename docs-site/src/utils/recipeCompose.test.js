const assert = require("node:assert/strict");
const test = require("node:test");
const yaml = require("js-yaml");

const { composeYaml } = require("./recipeCompose");

function fragment(label, obj) {
  return { label: label, text: yaml.dump(obj) };
}

test("merges named sections by name across files", () => {
  const recipe = fragment("workflow.yaml", {
    version: "0.1",
    variables: [{ name: "MODEL", value: "meta-llama/Llama" }],
    workflow: {
      name: "wf",
      tasks: [{ name: "vllm_server", operator: { name: "vllm_runtime" } }],
    },
  });
  const machine = fragment("machines/backend/k8s.yaml", {
    version: "0.1",
    variables: [{ name: "K8S_NAMESPACE", value: "" }],
    backends: [{ name: "compute", type: "kubernetes" }],
    operators: [{ name: "vllm_runtime", type: "k8s" }],
  });

  const out = yaml.load(composeYaml([recipe, machine]));

  assert.deepEqual(out.variables.map((v) => v.name).sort(), ["K8S_NAMESPACE", "MODEL"]);
  assert.equal(out.backends[0].name, "compute");
  assert.equal(out.operators[0].type, "k8s");
  assert.equal(out.workflow.tasks[0].operator.name, "vllm_runtime");
});

test("deep-merges same-name backend entries instead of replacing them", () => {
  const base = fragment("machines/backend/k8s.yaml", {
    version: "0.1",
    backends: [{ name: "compute", type: "kubernetes", namespace: "${{ variables.K8S_NAMESPACE }}" }],
  });
  const overlay = fragment("machines/backend/k8s_model_pvc.yaml", {
    version: "0.1",
    backends: [{ name: "compute", volumes: [{ name: "models", claim: "pvc" }] }],
  });

  const out = yaml.load(composeYaml([base, overlay]));

  assert.equal(out.backends.length, 1);
  assert.equal(out.backends[0].type, "kubernetes");
  assert.equal(out.backends[0].volumes[0].claim, "pvc");
});

test("a later fragment's task field replaces the earlier one wholesale, not merged", () => {
  const recipe = fragment("workflow.yaml", {
    version: "0.1",
    workflow: {
      name: "wf",
      tasks: [{ name: "vllm_server", probes: { readiness: { tcp_port: { port: 8000 } } } }],
    },
  });
  const machine = fragment("machines/backend/k8s.yaml", {
    version: "0.1",
    workflow: {
      tasks: [{ name: "vllm_server", probes: { readiness: [{ log_watch: { regex_pattern: "ok" } }] } }],
    },
  });

  const out = yaml.load(composeYaml([recipe, machine]));

  assert.ok(Array.isArray(out.workflow.tasks[0].probes.readiness));
  assert.equal(out.workflow.tasks[0].probes.readiness[0].log_watch.regex_pattern, "ok");
});

test("conflicting version across files throws", () => {
  const a = fragment("a.yaml", { version: "0.1", variables: [] });
  const b = fragment("b.yaml", { version: "0.2", variables: [] });
  assert.throws(() => composeYaml([a, b]), /Version conflict/);
});

test("drops null-valued keys from the composed output", () => {
  const a = fragment("a.yaml", { version: "0.1", variables: [{ name: "X", description: null, value: "1" }] });
  const out = yaml.load(composeYaml([a]));
  assert.equal("description" in out.variables[0], false);
});
