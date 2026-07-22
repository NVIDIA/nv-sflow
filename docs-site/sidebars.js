const path = require("path");
const fs = require("fs");
const { buildDocsSidebar } = require("./sidebarsConfig");

// Mirror docusaurus.config.js: prefer the generated develop snapshot, otherwise
// fall back to the repo-level docs/ directory for local dev.
function currentDocsPath() {
  const generated = path.resolve(__dirname, ".generated", "current-docs");
  return fs.existsSync(generated) ? generated : path.resolve(__dirname, "..", "docs");
}

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  docs: buildDocsSidebar(currentDocsPath()),
};

module.exports = sidebars;
