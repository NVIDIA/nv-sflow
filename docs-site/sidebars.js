const path = require("path");
const fs = require("fs");
const { buildDocsSidebar, buildAgentsSidebar } = require("./sidebarsConfig");

// Mirror docusaurus.config.js: prefer the generated develop snapshot, otherwise
// fall back to the repo-level docs/ directory for local dev.
function currentDocsPath() {
  const generated = path.resolve(__dirname, ".generated", "current-docs");
  return fs.existsSync(generated) ? generated : path.resolve(__dirname, "..", "docs");
}

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const currentDocs = currentDocsPath();
const agents = buildAgentsSidebar(currentDocs);

const sidebars = {
  docs: buildDocsSidebar(currentDocs),
  // Spread rather than assign: Docusaurus rejects an empty/undefined sidebar, so a
  // snapshot without skills must omit the key entirely.
  ...(agents ? { agents } : {}),
};

module.exports = sidebars;
