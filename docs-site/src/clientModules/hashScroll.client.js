const { createHashScroller } = require("../utils/hashScroll");

let scroller = null;

function getScroller() {
  if (typeof window === "undefined" || typeof document === "undefined") {
    return null;
  }
  if (!scroller) {
    scroller = createHashScroller({ window, document });
  }
  return scroller;
}

// Fires after every route render (including the initial load), which is after
// Docusaurus has done its own scroll handling, so this reliably wins the race
// for deep links opened in a new tab from the feature map.
export function onRouteDidUpdate() {
  const instance = getScroller();
  if (instance) {
    instance.scrollToHash();
  }
}

// Safety net for the very first paint in case onRouteDidUpdate has not fired yet.
const initial = getScroller();
if (initial) {
  initial.scrollToHash();
}
