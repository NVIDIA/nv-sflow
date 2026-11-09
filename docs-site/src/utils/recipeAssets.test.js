const assert = require("node:assert/strict");
const test = require("node:test");
const { gzipSync } = require("node:zlib");

function setup(t) {
  const before = { document: global.document, window: global.window };
  const scripts = [];
  global.document = {
    createElement: () => ({remove() { this.removed = true; }}),
    body: {appendChild: script => scripts.push(script)},
  };
  global.window = {};
  t.after(() => {
    for (const [key, value] of Object.entries(before)) {
      if (value === undefined) delete global[key]; else global[key] = value;
    }
  });
  delete require.cache[require.resolve("./recipeAssets")];
  return { scripts, ...require("./recipeAssets") };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test("index and builder load concurrently, share requests, and initialize only when both finish", async t => {
  const {scripts, loadRecipeScripts} = setup(t);
  let release, requests = 0, finished = false;
  t.mock.method(global, "fetch", () => {
    requests++;
    return new Promise(resolve => { release = resolve; });
  });
  const first = loadRecipeScripts("/index.gz", "/builder.js").then(() => { finished = true; });
  const second = loadRecipeScripts("/index.gz", "/builder.js");
  await tick();
  assert.equal(requests, 1);
  assert.deepEqual(scripts.map(s => s.src), ["/builder.js"]);
  scripts[0].onload();
  await tick();
  assert.equal(finished, false);
  const data = {RECIPE_DATA: [{model_group: "org/model"}], MACHINE_DATA: {"gpu.yaml": "version: '0.1'"}};
  release(new Response(gzipSync(JSON.stringify(data))));
  await Promise.all([first, second]);
  assert.deepEqual(global.window.RECIPE_DATA, data.RECIPE_DATA);
  assert.deepEqual(global.window.MACHINE_DATA, data.MACHINE_DATA);
  await loadRecipeScripts("/index.gz", "/builder.js");
  assert.equal(scripts.length, 1);
  assert.equal(requests, 1);
});

test("failed index retries while pending builder is reused, then failed builder retries alone", async t => {
  const {scripts, loadRecipeScripts} = setup(t);
  let requests = 0, finished = false;
  t.mock.method(global, "fetch", async () => {
    requests++;
    return requests === 1 ? new Response("missing", {status: 503}) :
      new Response(gzipSync(JSON.stringify({RECIPE_DATA: [], MACHINE_DATA: {}})));
  });
  await assert.rejects(loadRecipeScripts("/index.gz", "/builder.js"), /503/);
  const second = loadRecipeScripts("/index.gz", "/builder.js");
  const rejection = assert.rejects(second, /builder.js/).then(() => { finished = true; });
  // Wait for metadata to publish first; the builder is still pending.
  while (!global.window.RECIPE_DATA) await tick();
  assert.equal(finished, false);
  assert.equal(scripts.length, 1);
  assert.equal(requests, 2);
  scripts[0].onerror();
  await rejection;
  assert.equal(scripts[0].removed, true);
  const third = loadRecipeScripts("/index.gz", "/builder.js");
  await tick();
  assert.equal(requests, 2);
  assert.deepEqual(scripts.map(s => s.src), ["/builder.js", "/builder.js"]);
  scripts[1].onload();
  await third;
});

test("YAML retries HTTP and corrupt-gzip failures, shares pending work and caches only success", async t => {
  const {loadRecipeYaml} = setup(t);
  let requests = 0;
  t.mock.method(global, "fetch", async () => {
    requests++;
    if (requests === 1) return new Response("missing", {status: 503});
    if (requests === 2) return new Response("not gzip");
    return new Response(gzipSync(JSON.stringify({"workflow.yaml": "version: '0.1'"})));
  });
  await assert.rejects(loadRecipeYaml("/yaml.gz"), /503/);
  await assert.rejects(loadRecipeYaml("/yaml.gz"));
  assert.equal(global.window.RECIPE_YAML, undefined);
  await Promise.all([loadRecipeYaml("/yaml.gz"), loadRecipeYaml("/yaml.gz")]);
  await loadRecipeYaml("/yaml.gz");
  assert.equal(requests, 3);
  assert.deepEqual(global.window.RECIPE_YAML, {"workflow.yaml": "version: '0.1'"});
});
