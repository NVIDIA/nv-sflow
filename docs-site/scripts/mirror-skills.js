const fs = require("fs");
const path = require("path");

// Per-skill presentation metadata. Unknown skills still mirror, with a
// title-cased fallback label so new skills show up without code changes.
const SKILL_META = {
  "writing-sflow-yaml": { label: "Writing sflow YAML", position: 3 },
  "sflow-error-analysis": { label: "Error Analysis", position: 4 },
};
const AGENTS_GUIDELINES_POSITION = 5;
const SKIP_DIRS = new Set(["scripts", "__pycache__"]);

function stripFrontmatter(md) {
  if (!md.startsWith("---")) return md;
  const close = md.indexOf("\n---", 3);
  if (close === -1) return md;
  const nextLine = md.indexOf("\n", close + 1);
  return nextLine === -1 ? "" : md.slice(nextLine + 1).replace(/^\s+/, "");
}

function frontmatter(fields) {
  const body = Object.entries(fields)
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");
  return `---\n${body}\n---\n\n`;
}

function fallbackLabel(name) {
  return name
    .split(/[-_]/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function writeDoc(file, fields, body) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, frontmatter(fields) + body.replace(/\s*$/, "") + "\n");
}

// Mirror the packaged skills (src/sflow/skills) plus the hand-written overview /
// setup pages into a Docusaurus `agents/` docs directory. Returns the skill names
// that were mirrored. No-op (and no dir created) when the skills source is absent,
// so doc versions that predate the skills feature simply omit the section.
function mirrorSkillsToAgents({ skillsDir, handwrittenDir, destDir }) {
  if (!skillsDir || !fs.existsSync(skillsDir)) {
    return { skills: [] };
  }

  fs.rmSync(destDir, { recursive: true, force: true });
  fs.mkdirSync(destDir, { recursive: true });

  // Hand-written overview + setup pages (copied verbatim, they already carry frontmatter).
  if (handwrittenDir && fs.existsSync(handwrittenDir)) {
    for (const name of ["intro.md", "setup.md"]) {
      const src = path.join(handwrittenDir, name);
      if (fs.existsSync(src)) {
        fs.copyFileSync(src, path.join(destDir, name));
      }
    }
  }

  // AGENTS.md -> agents-guidelines.md (keep its H1 as the page title).
  const agentsMd = path.join(skillsDir, "AGENTS.md");
  if (fs.existsSync(agentsMd)) {
    writeDoc(
      path.join(destDir, "agents-guidelines.md"),
      { sidebar_position: AGENTS_GUIDELINES_POSITION, sidebar_label: "AGENTS.md guidelines" },
      stripFrontmatter(fs.readFileSync(agentsMd, "utf8")),
    );
  }

  // One Docusaurus category per skill directory.
  const skills = [];
  for (const entry of fs.readdirSync(skillsDir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory() || entry.name.startsWith("_") || SKIP_DIRS.has(entry.name)) continue;
    const skillSrc = path.join(skillsDir, entry.name);
    const skillMd = path.join(skillSrc, "SKILL.md");
    if (!fs.existsSync(skillMd)) continue;

    skills.push(entry.name);
    const meta = SKILL_META[entry.name] || { label: fallbackLabel(entry.name), position: 50 };
    const outDir = path.join(destDir, entry.name);
    fs.mkdirSync(outDir, { recursive: true });
    fs.writeFileSync(
      path.join(outDir, "_category_.json"),
      JSON.stringify({ label: meta.label, position: meta.position, collapsed: false }, null, 2) + "\n",
    );

    writeDoc(
      path.join(outDir, "index.md"),
      { sidebar_position: 1, sidebar_label: "Overview" },
      stripFrontmatter(fs.readFileSync(skillMd, "utf8")),
    );

    // Sibling markdown (schema-reference, examples, error-catalog, ...). Scripts are skipped.
    let position = 2;
    for (const file of fs.readdirSync(skillSrc).sort()) {
      if (file === "SKILL.md" || !file.endsWith(".md")) continue;
      writeDoc(
        path.join(outDir, file),
        { sidebar_position: position },
        fs.readFileSync(path.join(skillSrc, file), "utf8"),
      );
      position += 1;
    }
  }

  return { skills };
}

module.exports = { mirrorSkillsToAgents, stripFrontmatter };
