const assert = require("node:assert/strict");
const test = require("node:test");

const { createFrameEscaper } = require("./frameEscape");

function makeWindow({ framed, href = "https://pages.example.com/docs/user/intro", replaceThrows = false } = {}) {
  const calls = { replace: [], hrefSet: [] };
  const topLocation = {
    replace(url) {
      if (replaceThrows) throw new Error("SecurityError: cross-origin Location");
      calls.replace.push(url);
    },
    set href(url) {
      calls.hrefSet.push(url);
    },
    get href() {
      return "about:blank";
    },
  };
  const win = { location: { href } };
  win.self = win;
  win.top = framed ? { location: topLocation } : win;
  return { win, calls };
}

test("no-op when not framed (top === self)", () => {
  const { win, calls } = makeWindow({ framed: false });
  const escaper = createFrameEscaper({ window: win });
  assert.equal(escaper.isFramed(), false);
  assert.equal(escaper.escape(), false);
  assert.deepEqual(calls.replace, []);
  assert.deepEqual(calls.hrefSet, []);
});

test("escapes to the top window with the current href when framed", () => {
  const { win, calls } = makeWindow({ framed: true });
  const escaper = createFrameEscaper({ window: win });
  assert.equal(escaper.isFramed(), true);
  assert.equal(escaper.escape(), true);
  assert.deepEqual(calls.replace, ["https://pages.example.com/docs/user/intro"]);
  assert.deepEqual(calls.hrefSet, []);
});

test("falls back to assigning href when replace() throws (cross-origin top)", () => {
  const { win, calls } = makeWindow({ framed: true, replaceThrows: true, href: "https://pages.example.com/docs/x" });
  const escaper = createFrameEscaper({ window: win });
  assert.equal(escaper.escape(), true);
  assert.deepEqual(calls.replace, []);
  assert.deepEqual(calls.hrefSet, ["https://pages.example.com/docs/x"]);
});

test("no window => safe no-op", () => {
  const escaper = createFrameEscaper({});
  assert.equal(escaper.isFramed(), false);
  assert.equal(escaper.escape(), false);
});
