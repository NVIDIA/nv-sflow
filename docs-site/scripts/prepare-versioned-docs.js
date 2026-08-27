#!/usr/bin/env node

const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const { buildDocsSidebar } = require("../sidebarsConfig");
const { mirrorSkillsToAgents } = require("./mirror-skills");

const DOCS_SITE_DIR = path.resolve(__dirname, "..");
const REPO_ROOT = path.resolve(DOCS_SITE_DIR, "..");
const GENERATED_CURRENT_DIR = path.join(DOCS_SITE_DIR, ".generated", "current-docs");
const VERSIONED_DOCS_DIR = path.join(DOCS_SITE_DIR, "versioned_docs");
const VERSIONED_SIDEBARS_DIR = path.join(DOCS_SITE_DIR, "versioned_sidebars");
const VERSIONS_JSON = path.join(DOCS_SITE_DIR, "versions.json");
const SKILLS_REPO_PATH = "src/sflow/skills";

function docsPaths({ repoRoot = REPO_ROOT, docsSiteDir = DOCS_SITE_DIR } = {}) {
  return {
    repoRoot,
    docsSiteDir,
    agentsSrcDir: path.join(docsSiteDir, "agents-src"),
    generatedCurrentDir: path.join(docsSiteDir, ".generated", "current-docs"),
    versionedDocsDir: path.join(docsSiteDir, "versioned_docs"),
    versionedSidebarsDir: path.join(docsSiteDir, "versioned_sidebars"),
    versionsJson: path.join(docsSiteDir, "versions.json"),
  };
}

function isReleaseTag(ref) {
  return /^v\d+\.\d+\.\d+$/.test(ref);
}

function safeVersionDirName(label) {
  return `version-${label.replace(/[^A-Za-z0-9._-]/g, "_")}`;
}

function semverParts(tag) {
  return tag
    .replace(/^v/, "")
    .split(".")
    .map((part) => Number.parseInt(part, 10));
}

function compareReleaseTagsDesc(a, b) {
  const aa = semverParts(a);
  const bb = semverParts(b);
  for (let i = 0; i < Math.max(aa.length, bb.length); i += 1) {
    const delta = (bb[i] || 0) - (aa[i] || 0);
    if (delta !== 0) return delta;
  }
  return a.localeCompare(b);
}

function buildDocVersionPlan({
  branches,
  tags,
  currentRef = "develop",
  currentLabel = "develop",
  currentSource = currentDocsSource(),
}) {
  const branchSet = new Set(branches);
  const versioned = [];

  if (branchSet.has("main") && currentRef !== "main") {
    versioned.push({ label: "main", ref: "main", source: "gitRef" });
  }

  const releaseVersions = [...new Set(tags)]
    .filter(isReleaseTag)
    .sort(compareReleaseTagsDesc)
    .map((tag) => ({ label: tag, ref: tag, source: "gitRef" }));

  versioned.push(...releaseVersions);

  return {
    current: { label: currentLabel, ...currentSource, ref: currentSource.ref || currentRef },
    versioned,
    versionsJson: versioned.map((version) => version.label),
  };
}

function currentDocsSource(env = process.env) {
  if (env.GITLAB_CI || env.GITHUB_ACTIONS) {
    return { ref: "origin/develop", source: "gitRef" };
  }
  return { ref: "develop", source: "workingTree" };
}

function gitRefExists(ref) {
  try {
    run("git", ["rev-parse", "--verify", "--quiet", `${ref}^{commit}`]);
    return true;
  } catch {
    return false;
  }
}

function resolveGitRef(label) {
  if (gitRefExists(label)) return label;
  const remoteRef = `origin/${label}`;
  if (gitRefExists(remoteRef)) return remoteRef;
  throw new Error(`Could not resolve git ref for docs version '${label}'`);
}

function run(command, args, options = {}) {
  return childProcess.execFileSync(command, args, {
    cwd: REPO_ROOT,
    encoding: "utf8",
    stdio: options.stdio || ["ignore", "pipe", "pipe"],
  });
}

function listBranches() {
  return ["develop", "main"].filter((branch) => {
    try {
      resolveGitRef(branch);
      return true;
    } catch {
      return false;
    }
  });
}

function listReleaseTags() {
  const out = run("git", ["tag", "--list", "v[0-9]*.[0-9]*.[0-9]*"]);
  return out
    .split("\n")
    .map((line) => line.trim())
    .filter(Boolean);
}

function emptyDir(dir) {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(dir, { recursive: true });
}

// Extract a tar stream (from `git archive`) into `dest`. Runs tar with cwd:dest
// rather than passing `-C dest`, because GNU tar (MSYS/Git-Bash) mangles Windows
// drive-letter paths like `C:\...` when they arrive as an argument; Node sets the
// child's cwd through the OS instead. `dest` must already exist (both callers
// create it first). Returns the spawnSync result so callers check `.status`.
function extractTar(input, dest, stripComponents) {
  return childProcess.spawnSync("tar", ["-x", `--strip-components=${stripComponents}`], {
    cwd: dest,
    input,
    stdio: ["pipe", "inherit", "inherit"],
  });
}

function copyDocsFromRef(ref, destination, { repoRoot = REPO_ROOT } = {}) {
  emptyDir(destination);
  const archive = childProcess.spawnSync(
    "git",
    ["archive", "--format=tar", ref, "docs"],
    {
      cwd: repoRoot,
      stdio: ["ignore", "pipe", "inherit"],
    },
  );
  if (archive.status !== 0) {
    throw new Error(`Failed to archive docs from ${ref}`);
  }
  const extract = extractTar(archive.stdout, destination, 1);
  if (extract.status !== 0) {
    throw new Error(`Failed to extract docs from ${ref}`);
  }
}

function copyDocsFromWorkingTree(destination, { repoRoot = REPO_ROOT } = {}) {
  const source = path.join(repoRoot, "docs");
  emptyDir(destination);
  fs.cpSync(source, destination, { recursive: true });
}

// Extract the packaged skills tree (src/sflow/skills) from a git ref into a temp
// directory so each doc version mirrors the skills as they existed at that ref.
// Returns null when the ref has no skills (older versions), so the mirror is skipped.
function extractSkillsFromRef(ref, { repoRoot = REPO_ROOT } = {}) {
  const archive = childProcess.spawnSync(
    "git",
    ["archive", "--format=tar", ref, SKILLS_REPO_PATH],
    { cwd: repoRoot, stdio: ["ignore", "pipe", "pipe"] },
  );
  if (archive.status !== 0) return null;
  const dest = fs.mkdtempSync(path.join(os.tmpdir(), "sflow-skills-"));
  const extract = extractTar(archive.stdout, dest, 3);
  if (extract.status !== 0) {
    fs.rmSync(dest, { recursive: true, force: true });
    return null;
  }
  return dest;
}

function currentSkillsDir(plan, paths) {
  if (plan.current.source === "workingTree") {
    return path.join(paths.repoRoot, SKILLS_REPO_PATH);
  }
  return extractSkillsFromRef(plan.current.ref, paths);
}

function writeJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`);
}

// In-page anchors that were already broken in a published snapshot. A released tag
// cannot be re-cut, so for frozen versions the only place left to repair them is
// here, at extraction time. Current docs are deliberately NOT covered: those are
// fixed at source in docs/.
//
// Each entry carries the heading it assumes, and the repair only fires when that
// heading is actually present. `main` is a branch, not a frozen tag -- once the
// source fix lands there its snapshot already has the corrected heading, and an
// unconditional rewrite would break the link a second time in the other direction.
const FROZEN_ANCHOR_REPAIRS = [
  {
    from: "#modular-inference-recipe-inference_x_v2",
    to: "#inference_x_v2",
    onlyWhenHeading: /^### inference_x_v2\s*$/m,
  },
];

function repairFrozenAnchors(text) {
  let out = text;
  for (const repair of FROZEN_ANCHOR_REPAIRS) {
    if (out.includes(repair.from) && repair.onlyWhenHeading.test(out)) {
      out = out.replaceAll(repair.from, repair.to);
    }
  }
  return out;
}

function rewriteVersionedDocsLinks(rootDir, versionLabel) {
  const entries = fs.readdirSync(rootDir, { withFileTypes: true });
  for (const entry of entries) {
    const entryPath = path.join(rootDir, entry.name);
    if (entry.isDirectory()) {
      rewriteVersionedDocsLinks(entryPath, versionLabel);
    } else if (entry.isFile() && entry.name.endsWith(".md")) {
      const text = fs.readFileSync(entryPath, "utf8");
      const rewritten = repairFrozenAnchors(
        text.replaceAll("](/docs/", `](/docs/${versionLabel}/`),
      );
      if (rewritten !== text) {
        fs.writeFileSync(entryPath, rewritten);
      }
    }
  }
}

function prepareVersionedDocs(plan, options = {}) {
  const paths = docsPaths(options);
  if (plan.current.source === "workingTree") {
    copyDocsFromWorkingTree(paths.generatedCurrentDir, paths);
  } else {
    copyDocsFromRef(plan.current.ref, paths.generatedCurrentDir, paths);
  }
  // Mirror the develop/current skills into the current docs as an agents/ section.
  mirrorSkillsToAgents({
    skillsDir: currentSkillsDir(plan, paths),
    handwrittenDir: paths.agentsSrcDir,
    destDir: path.join(paths.generatedCurrentDir, "agents"),
  });
  emptyDir(paths.versionedDocsDir);
  emptyDir(paths.versionedSidebarsDir);

  for (const version of plan.versioned) {
    const versionDir = path.join(paths.versionedDocsDir, safeVersionDirName(version.label));
    copyDocsFromRef(
      version.ref,
      versionDir,
      paths,
    );
    rewriteVersionedDocsLinks(versionDir, version.label);
    // Mirror that version's skills before building its sidebar so the
    // "Agent Skills" category is included when the version ships skills.
    mirrorSkillsToAgents({
      skillsDir: extractSkillsFromRef(version.ref, paths),
      handwrittenDir: paths.agentsSrcDir,
      destDir: path.join(versionDir, "agents"),
    });
    writeJson(
      path.join(paths.versionedSidebarsDir, `${safeVersionDirName(version.label)}-sidebars.json`),
      { docs: buildDocsSidebar(versionDir) },
    );
  }

  writeJson(paths.versionsJson, plan.versionsJson);
}

function main() {
  const branches = listBranches();
  const tags = listReleaseTags();
  const plan = buildDocVersionPlan({ branches, tags });
  if (plan.current.source === "gitRef") {
    plan.current.ref = resolveGitRef(plan.current.ref);
  }
  plan.versioned = plan.versioned.map((version) => ({
    ...version,
    ref: isReleaseTag(version.label) ? version.ref : resolveGitRef(version.label),
  }));
  prepareVersionedDocs(plan);
  console.log(
    `Prepared docs versions: current=${plan.current.label}; versions=${plan.versionsJson.join(", ") || "(none)"}`,
  );
}

if (require.main === module) {
  main();
}

module.exports = {
  buildDocVersionPlan,
  repairFrozenAnchors,
  currentDocsSource,
  docsPaths,
  isReleaseTag,
  safeVersionDirName,
  resolveGitRef,
  prepareVersionedDocs,
};
