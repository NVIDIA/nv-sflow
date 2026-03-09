// @ts-nocheck

const path = require("path");
const fs = require("fs");

function readReleasedDocVersions() {
  try {
    const p = path.resolve(__dirname, "versions.json");
    return JSON.parse(fs.readFileSync(p, "utf8"));
  } catch {
    return [];
  }
}

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "sflow",
  tagline: "Workflow Orchestrator with Pluggable Backends",
  url: "https://urban-carnival-zg812kp.pages.github.io",
  baseUrl: "/",
  trailingSlash: false,
  favicon: "img/sflow-logo.ico",
  onBrokenLinks: "throw",
  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: "warn",
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
          // Reuse the existing markdown under repo-level docs/
          path: path.resolve(__dirname, "..", "docs"),
          routeBasePath: "docs",
          sidebarPath: require.resolve("./sidebars.js"),
          // Keep current (unreleased) docs at /docs/... so existing links don't break.
          // Released versions will live under /docs/<version>/...
          lastVersion: "current",
          versions: {
            current: {
              label: "dev",
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
      "@docusaurus/plugin-client-redirects",
      {
        // Add a stable alias path for "dev" so /docs/dev/... redirects to /docs/...
        // This is useful for sharing links that always target the latest docs.
        createRedirects(existingPath) {
          // Only alias CURRENT docs, not released versions (e.g. /docs/0.1/...).
          const released = new Set(readReleasedDocVersions());
          const parts = existingPath.split("/").filter(Boolean); // ["docs", ...]
          if (parts[0] !== "docs") return undefined;
          if (parts.length >= 2 && released.has(parts[1])) return undefined;

          // Alias /docs/<...> => /docs/dev/<...>
          return [existingPath.replace(/^\/docs(\/|$)/, "/docs/dev$1")];
        },
        // /docs is not a real route by default; redirect to an existing doc page.
        redirects: [{ from: "/docs/dev", to: "/docs/user/intro" }],
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
        copyright: `Copyright © ${new Date().getFullYear()} sflow`,
      },
    }),
};

module.exports = config;
