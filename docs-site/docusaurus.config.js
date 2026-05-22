// @ts-nocheck

const path = require("path");
const fs = require("fs");

function readVersionedDocIds() {
  try {
    const p = path.resolve(__dirname, "versions.json");
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return [];
  }
}

function currentDocsPath() {
  const generated = path.resolve(__dirname, ".generated", "current-docs");
  return fs.existsSync(generated) ? generated : path.resolve(__dirname, "..", "docs");
}

const baseUrl = process.env.DOCS_BASE_URL || "/";

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "NV-sflow",
  tagline: "Declarative Workflow Descriptor with Swappable Backends",
  url: "https://nvidia.github.io",
  baseUrl,
  trailingSlash: false,
  favicon: "img/sflow-logo.ico",
  onBrokenLinks: "throw",
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
      onBrokenMarkdownImages: "warn",
    },
  },
  organizationName: "NVIDIA",
  projectName: "nv-sflow",
  deploymentBranch: "gh-pages",
  themes: ["@docusaurus/theme-mermaid"],

  presets: [
    [
      "classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          // Builds use generated docs from the develop branch. Local dev falls
          // back to the repo-level docs/ directory if generation has not run.
          path: currentDocsPath(),
          routeBasePath: "docs",
          sidebarPath: require.resolve("./sidebars.js"),
          // Keep develop docs at /docs/... so existing links don't break.
          // Main and released tags live under /docs/<version>/...
          lastVersion: "current",
          versions: {
            current: {
              label: "develop",
              banner: "none",
            },
            main: {
              label: "main",
              banner: "none",
            },
          },
          showLastUpdateAuthor: false,
          showLastUpdateTime: false,
        },
        blog: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  plugins: [
    [
      require.resolve("@cmfcmf/docusaurus-search-local"),
      {
        indexBlog: false,
      },
    ],
    [
      "@docusaurus/plugin-client-redirects",
      {
        // Add a stable alias path for "develop" so /docs/develop/... redirects to /docs/...
        // This is useful for sharing links that explicitly target develop docs.
        createRedirects(existingPath) {
          // Only alias CURRENT docs, not versioned docs (e.g. /docs/main/...).
          const versioned = new Set(readVersionedDocIds());
          const parts = existingPath.split("/").filter(Boolean); // ["docs", ...]
          if (parts[0] !== "docs") return undefined;
          if (parts.length >= 2 && versioned.has(parts[1])) return undefined;

          // Alias /docs/<...> => /docs/develop/<...>
          return [existingPath.replace(/^\/docs(\/|$)/, "/docs/develop$1")];
        },
        redirects: [
          // /docs is not a real route by default; redirect to an existing doc page.
          { from: "/docs/develop", to: "/docs/user/intro" },
        ],
      },
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: "dark",
        disableSwitch: false,
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: "sflow",
        logo: {
          alt: "sflow",
          src: "img/sflow-logo.jpg",
        },
        items: [
          { type: "doc", docId: "user/intro", label: "Docs", position: "left" },
          { type: "search", position: "left" },
          { type: "docsVersionDropdown", position: "right" },
          {
            href: "https://github.com/NVIDIA/nv-sflow",
            label: "GitHub",
            "aria-label": "GitHub repository",
            className: "header-github-link",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          {
            title: "Docs",
            items: [
              { label: "User guide", to: "/docs/user/intro" },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} NVIDIA Corporation. Licensed under Apache 2.0.`,
      },
    }),
};

module.exports = config;
