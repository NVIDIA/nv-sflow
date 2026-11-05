const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const {gzipSync} = require("node:zlib");

function compressRecipeIndex(source) {
  // The source is the trusted, checked-in upstream export, not user input.
  const json = vm.runInNewContext(source + ";JSON.stringify({RECIPE_DATA, MACHINE_DATA})");
  return gzipSync(json, {level: 9});
}

if (require.main === module) {
  const dir = path.resolve(__dirname, "../static/vllm-recipes");
  const source = fs.readFileSync(path.join(dir, "recipe_data.js"), "utf8");
  fs.writeFileSync(path.join(dir, "recipe_data.json.gz"), compressRecipeIndex(source));
}

module.exports = {compressRecipeIndex};
