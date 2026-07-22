// Pop the docs app out of a wrapping frame so per-page URLs stay shareable.
//
// Why this exists: when the built site is served inside a frame (e.g. a hosting proxy or
// portal that wraps the GitLab Pages deployment), the Docusaurus SPA's history.pushState
// updates the *framed* document's location, not the top browser address bar. Navigation
// still works, but the URL in the address bar never changes — so a reader can't copy/share
// a link to the page (or section) they're on. Breaking out to the top window restores a
// real, shareable URL per page.
//
// Safe by construction:
//   * No-op when not framed (local `npm run serve`, direct GitLab Pages) — top === self.
//   * The site's own homepage / feature-map embed STATIC html (sflow_intro.html /
//     feature-map-embed.html) in an iframe; those files never load this Docusaurus client module,
//     so this only ever fires when the Docusaurus APP itself is unexpectedly framed.
//   * Uses location.replace so the wrapper URL is not left in the back/forward history.
function createFrameEscaper({ window: windowObj } = {}) {
  function isFramed() {
    if (!windowObj) return false;
    try {
      return windowObj.top !== windowObj.self;
    } catch (_) {
      // Reading window.top threw (a cross-origin ancestor blocked access) => we are framed.
      return true;
    }
  }

  function escape() {
    if (!windowObj || !isFramed()) return false;
    const href = windowObj.location && windowObj.location.href;
    if (!href) return false;
    // Navigating the top window is permitted even across origins (it's a write, not a
    // cross-origin read). replace() is cleaner but throws on a cross-origin Location, so
    // fall back to assigning href, which is always allowed.
    try {
      windowObj.top.location.replace(href);
      return true;
    } catch (_) {
      try {
        windowObj.top.location.href = href;
        return true;
      } catch (__) {
        return false;
      }
    }
  }

  return { isFramed, escape };
}

module.exports = { createFrameEscaper };
