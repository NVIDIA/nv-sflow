// Hands the vLLM Recipes page its logic modules.
//
// The rules live in plain CommonJS modules so node --test can exercise them
// (src/utils/recipeSelection.test.js, recipeCommand.test.js); the page's
// builder.js is a classic script in static/ that webpack never processes, so
// it cannot import them. A client module is the seam between the two: it is
// bundled (so `require` works) and it runs in the browser (so it can put the
// functions somewhere a plain script can reach) - the same arrangement the
// page already uses for RECIPE_DATA.
const { resolveSelection } = require("../utils/recipeSelection");
const { planCommand, renderCommand, basename } = require("../utils/recipeCommand");
const { composeYaml } = require("../utils/recipeCompose");

function publish() {
  if (typeof window !== "undefined") {
    window.__vllmRecipes = { resolveSelection, planCommand, renderCommand, basename, composeYaml };
  }
}

publish();

export function onRouteDidUpdate() {
  publish();
}
