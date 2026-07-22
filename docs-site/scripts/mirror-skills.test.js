const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("fs");
const os = require("os");
const path = require("path");

const { mirrorSkillsToAgents } = require("./mirror-skills");

function write(file, content) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, content);
}

function makeFixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "sflow-mirror-"));
  const skills = path.join(root, "skills");
  const handwritten = path.join(root, "handwritten");
  const dest = path.join(root, "agents");

  write(path.join(skills, "AGENTS.md"), "# sflow Agent Guidelines\n\nDo the thing.\n");
  write(
    path.join(skills, "writing-sflow-yaml", "SKILL.md"),
    "---\nname: writing-sflow-yaml\ndescription: write yaml\n---\n\n# Writing sflow YAML Configurations\n\nSee [schema-reference.md](schema-reference.md).\n",
  );
  write(path.join(skills, "writing-sflow-yaml", "schema-reference.md"), "# Schema reference\n");
  write(path.join(skills, "writing-sflow-yaml", "examples.md"), "# Examples\n");
  write(path.join(skills, "writing-sflow-yaml", "scripts", "validate_sflow_yaml.py"), "print('x')\n");
  write(
    path.join(skills, "sflow-error-analysis", "SKILL.md"),
    "---\nname: sflow-error-analysis\ndescription: debug\n---\n\n# sflow Error Analysis\n",
  );
  write(path.join(skills, "sflow-error-analysis", "error-catalog.md"), "# Error catalog\n");

  write(path.join(handwritten, "intro.md"), "---\nsidebar_position: 1\n---\n\n# Agent Skills\n");
  write(path.join(handwritten, "setup.md"), "---\nsidebar_position: 2\n---\n\n# Setup\n");

  return { root, skills, handwritten, dest };
}

test("mirrors handwritten pages, AGENTS.md, and each skill into versioned doc pages", () => {
  const fx = makeFixture();
  const result = mirrorSkillsToAgents({ skillsDir: fx.skills, handwrittenDir: fx.handwritten, destDir: fx.dest });

  assert.deepEqual(result.skills, ["sflow-error-analysis", "writing-sflow-yaml"]);

  // Hand-written pages copied verbatim.
  assert.ok(fs.existsSync(path.join(fx.dest, "intro.md")));
  assert.ok(fs.existsSync(path.join(fx.dest, "setup.md")));

  // AGENTS.md -> agents-guidelines.md with Docusaurus frontmatter, original heading preserved.
  const guidelines = fs.readFileSync(path.join(fx.dest, "agents-guidelines.md"), "utf8");
  assert.match(guidelines, /sidebar_position:\s*5/);
  assert.match(guidelines, /# sflow Agent Guidelines/);

  // Skill category metadata.
  const cat = JSON.parse(fs.readFileSync(path.join(fx.dest, "writing-sflow-yaml", "_category_.json"), "utf8"));
  assert.equal(cat.label, "Writing sflow YAML");
  assert.equal(cat.position, 3);

  // Skill index from SKILL.md: original frontmatter stripped, new frontmatter + heading kept.
  const idx = fs.readFileSync(path.join(fx.dest, "writing-sflow-yaml", "index.md"), "utf8");
  assert.match(idx, /sidebar_position:\s*1/);
  assert.doesNotMatch(idx, /name:\s*writing-sflow-yaml/);
  assert.match(idx, /# Writing sflow YAML Configurations/);
  assert.match(idx, /\[schema-reference\.md\]\(schema-reference\.md\)/);

  // Sibling docs copied; python scripts skipped.
  assert.ok(fs.existsSync(path.join(fx.dest, "writing-sflow-yaml", "schema-reference.md")));
  assert.ok(fs.existsSync(path.join(fx.dest, "writing-sflow-yaml", "examples.md")));
  assert.ok(!fs.existsSync(path.join(fx.dest, "writing-sflow-yaml", "scripts")));

  const cat2 = JSON.parse(fs.readFileSync(path.join(fx.dest, "sflow-error-analysis", "_category_.json"), "utf8"));
  assert.equal(cat2.label, "Error Analysis");
  assert.equal(cat2.position, 4);
  assert.ok(fs.existsSync(path.join(fx.dest, "sflow-error-analysis", "index.md")));
  assert.ok(fs.existsSync(path.join(fx.dest, "sflow-error-analysis", "error-catalog.md")));
});

test("does nothing when the skills source is absent", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "sflow-mirror-none-"));
  const dest = path.join(root, "agents");
  const result = mirrorSkillsToAgents({
    skillsDir: path.join(root, "missing-skills"),
    handwrittenDir: path.join(root, "missing-handwritten"),
    destDir: dest,
  });
  assert.deepEqual(result.skills, []);
  assert.ok(!fs.existsSync(dest));
});
