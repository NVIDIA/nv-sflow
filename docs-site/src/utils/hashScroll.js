function decodeHash(hash) {
  if (!hash || hash === "#") return "";

  const raw = hash.startsWith("#") ? hash.slice(1) : hash;
  try {
    return decodeURIComponent(raw);
  } catch {
    return raw;
  }
}

// Attempts are retried over ~1s so the target survives Docusaurus hydration,
// late content (web fonts, code blocks) reflow, and the router's own scroll
// handling on a fresh page load opened from the feature map.
const DEFAULT_ATTEMPT_DELAYS = [0, 80, 160, 280, 450, 700, 1000];

function createHashScroller({
  window: windowObj,
  document: documentObj,
  attemptDelays = DEFAULT_ATTEMPT_DELAYS,
} = {}) {
  let timers = [];
  let abort = null;

  function clearTimers() {
    if (windowObj) {
      timers.forEach((id) => windowObj.clearTimeout(id));
    }
    timers = [];
  }

  function removeAbortListeners() {
    if (!abort) return;
    if (windowObj) {
      windowObj.removeEventListener("wheel", abort.handler, abort.opts);
      windowObj.removeEventListener("touchstart", abort.handler, abort.opts);
      windowObj.removeEventListener("keydown", abort.handler);
    }
    abort = null;
  }

  function stop() {
    clearTimers();
    removeAbortListeners();
  }

  function scrollToHash() {
    if (!windowObj || !documentObj) return;

    // Cancel any in-flight run before starting a new one (e.g. rapid hash changes).
    stop();

    const id = decodeHash(windowObj.location.hash);
    if (!id) return;

    // Stop fighting the user the moment they take over scrolling.
    const handler = () => stop();
    const opts = { passive: true };
    abort = { handler, opts };
    windowObj.addEventListener("wheel", handler, opts);
    windowObj.addEventListener("touchstart", handler, opts);
    windowObj.addEventListener("keydown", handler);

    attemptDelays.forEach((delay, index) => {
      const timerId = windowObj.setTimeout(() => {
        const target = documentObj.getElementById(id);
        if (target) {
          target.scrollIntoView({ block: "start" });
        }
        if (index === attemptDelays.length - 1) {
          removeAbortListeners();
        }
      }, delay);
      timers.push(timerId);
    });
  }

  function install() {
    if (!windowObj) return () => {};

    scrollToHash();
    windowObj.addEventListener("hashchange", scrollToHash);

    return () => {
      windowObj.removeEventListener("hashchange", scrollToHash);
      stop();
    };
  }

  return { scrollToHash, install, stop };
}

module.exports = { createHashScroller };
