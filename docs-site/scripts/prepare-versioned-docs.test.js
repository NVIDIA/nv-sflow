const test = require("node:test");
const assert = require("node:assert/strict");
const childProcess = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const {
  buildDocVersionPlan,
  isReleaseTag,
  safeVersionDirName,
  currentDocsSource,
  prepareVersionedDocs,
} = require("./prepare-versioned-docs");

function git(cwd, args) {
  return childProcess.execFileSync("git", args, {
    cwd,
    encoding: "utf8",
    stdio: ["ignore", "pipe", "pipe"],
  });
}

function writeFile(filePath, content) {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
}

function commitAll(repo, message) {
  git(repo, ["add", "."]);
  git(repo, [
    "-c",
    "user.name=sflow test",
    "-c",
    "user.email=sflow-test@example.com",
    "commit",
    "-m",
    message,
  ]);
}

test("isReleaseTag accepts stable semantic release tags only", () => {
  assert.equal(isReleaseTag("v0.2.2"), true);
  assert.equal(isReleaseTag("v10.20.30"), true);
  assert.equal(isReleaseTag("v0.2.2-rc.1"), false);
  assert.equal(isReleaseTag("feature/foo"), false);
});

test("buildDocVersionPlan keeps develop as current and versions main plus release tags", () => {
  const plan = buildDocVersionPlan({
    branches: ["develop", "main", "feature/foo"],
    tags: ["v0.2.0", "test-tag", "v0.2.2", "v0.2.1-rc.1"],
    // Pin the current source so this case is deterministic regardless of the
    // ambient CI env (the env-dependent default is covered by the
    // currentDocsSource test below).
    currentSource: { ref: "develop", source: "workingTree" },
  });

  assert.deepEqual(plan.current, {
    label: "develop",
    ref: "develop",
    source: "workingTree",
  });
  assert.deepEqual(plan.versioned, [
    { label: "main", ref: "main", source: "gitRef" },
    { label: "v0.2.2", ref: "v0.2.2", source: "gitRef" },
    { label: "v0.2.0", ref: "v0.2.0", source: "gitRef" },
  ]);
  assert.deepEqual(plan.versionsJson, ["main", "v0.2.2", "v0.2.0"]);
});

test("safeVersionDirName maps version labels to Docusaurus directory names", () => {
  assert.equal(safeVersionDirName("main"), "version-main");
  assert.equal(safeVersionDirName("v0.2.2"), "version-v0.2.2");
  assert.equal(safeVersionDirName("release/foo"), "version-release_foo");
});

test("currentDocsSource uses local docs outside CI and origin develop in hosted CI", () => {
  assert.deepEqual(currentDocsSource({}), {
    ref: "develop",
    source: "workingTree",
  });
  assert.deepEqual(currentDocsSource({ GITLAB_CI: "true" }), {
    ref: "origin/develop",
    source: "gitRef",
  });
  assert.deepEqual(currentDocsSource({ GITHUB_ACTIONS: "true" }), {
    ref: "origin/develop",
    source: "gitRef",
  });
});

test("prepareVersionedDocs materializes current docs, versioned docs, sidebars, and versions.json", () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "sflow-docs-versions-"));
  const repo = path.join(tmp, "repo");
  const docsSite = path.join(repo, "docs-site");
  fs.mkdirSync(repo, { recursive: true });
  git(repo, ["init", "-b", "develop"]);

  writeFile(path.join(repo, "docs", "user", "intro.md"), "# Develop committed\n");
  commitAll(repo, "develop docs");

  git(repo, ["checkout", "-b", "main"]);
  writeFile(path.join(repo, "docs", "user", "intro.md"), "# Main docs\n");
  commitAll(repo, "main docs");

  git(repo, ["checkout", "develop"]);
  writeFile(path.join(repo, "docs", "user", "intro.md"), "# Release docs\n");
  writeFile(path.join(repo, "docs", "plc", "sflow_srd.md"), "See [SPP](/docs/sflow_spp).\n");
  commitAll(repo, "release docs");
  git(repo, ["tag", "v1.2.3"]);

  writeFile(path.join(repo, "docs", "user", "intro.md"), "# Local working docs\n");

  const plan = {
    current: { label: "develop", ref: "develop", source: "workingTree" },
    versioned: [
      { label: "main", ref: "main", source: "gitRef" },
      { label: "v1.2.3", ref: "v1.2.3", source: "gitRef" },
    ],
    versionsJson: ["main", "v1.2.3"],
  };

  prepareVersionedDocs(plan, { repoRoot: repo, docsSiteDir: docsSite });

  assert.equal(
    fs.readFileSync(path.join(docsSite, ".generated", "current-docs", "user", "intro.md"), "utf8"),
    "# Local working docs\n",
  );
  assert.equal(
    fs.readFileSync(path.join(docsSite, "versioned_docs", "version-main", "user", "intro.md"), "utf8"),
    "# Main docs\n",
  );
  assert.equal(
    fs.readFileSync(path.join(docsSite, "versioned_docs", "version-v1.2.3", "user", "intro.md"), "utf8"),
    "# Release docs\n",
  );
  assert.deepEqual(
    JSON.parse(fs.readFileSync(path.join(docsSite, "versions.json"), "utf8")),
    ["main", "v1.2.3"],
  );
  assert.deepEqual(
    JSON.parse(
      fs.readFileSync(
        path.join(docsSite, "versioned_sidebars", "version-v1.2.3-sidebars.json"),
        "utf8",
      ),
    ),
    {
      docs: [
        {
          type: "category",
          label: "Sflow User Guide",
          collapsed: false,
          items: [
            { type: "category", label: "Getting Started", collapsed: false, items: ["user/intro"] },
          ],
        },
      ],
    },
  );
  assert.equal(
    fs.readFileSync(path.join(docsSite, "versioned_docs", "version-v1.2.3", "plc", "sflow_srd.md"), "utf8"),
    "See [SPP](/docs/v1.2.3/sflow_spp).\n",
  );
});
