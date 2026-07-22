const { createFrameEscaper } = require("../utils/frameEscape");

let escaper = null;

function getEscaper() {
  if (typeof window === "undefined") {
    return null;
  }
  if (!escaper) {
    escaper = createFrameEscaper({ window });
  }
  return escaper;
}

// Break out on the very first load, before any client-side navigation, so a deep link
// opened inside a wrapping frame immediately shows its real URL in the address bar.
const initial = getEscaper();
if (initial) {
  initial.escape();
}

// Safety net: also check after every route change, in case the app is only framed on
// certain entry points. Once escaped, the app runs at top (not framed) so this is a no-op.
export function onRouteDidUpdate() {
  const instance = getEscaper();
  if (instance) {
    instance.escape();
  }
}
