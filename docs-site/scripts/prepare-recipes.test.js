const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const vm = require("node:vm");
const {gunzipSync} = require("node:zlib");
const {compressRecipeIndex} = require("./prepare-recipes");

test("compressed index preserves every recipe and machine string with a small transfer size", () => {
  const source = fs.readFileSync(require.resolve("../static/vllm-recipes/recipe_data.js"), "utf8");
  const expected = JSON.parse(vm.runInNewContext(source + ";JSON.stringify({RECIPE_DATA, MACHINE_DATA})"));
  const compressed = compressRecipeIndex(source);
  assert.deepEqual(JSON.parse(gunzipSync(compressed)), expected);
  assert.ok(compressed.length < Buffer.byteLength(source) / 5);
  assert.deepEqual(compressRecipeIndex(source), compressed, "repeated builds must be deterministic");
});
