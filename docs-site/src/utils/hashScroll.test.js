const assert = require("node:assert/strict");
const test = require("node:test");

const { createHashScroller } = require("./hashScroll");

function makeEnv(hash) {
  const tasks = [];
  const listeners = {};
  let nextId = 1;
  const scrolls = [];

  const window = {
    location: { hash },
    setTimeout(fn, ms) {
      const id = nextId++;
      tasks.push({ id, fn, ms, cancelled: false });
      return id;
    },
    clearTimeout(id) {
      const t = tasks.find((x) => x.id === id);
      if (t) t.cancelled = true;
    },
    addEventListener(type, fn) {
      (listeners[type] = listeners[type] || []).push(fn);
    },
    removeEventListener(type, fn) {
      if (!listeners[type]) return;
      listeners[type] = listeners[type].filter((f) => f !== fn);
    },
  };

  const document = {
    getElementById(id) {
      return {
        id,
        scrollIntoView(options) {
          scrolls.push({ id, options });
        },
      };
    },
  };

  function runPending() {
    for (let i = 0; i < tasks.length; i += 1) {
      const t = tasks[i];
      if (t.cancelled) continue;
      t.cancelled = true;
      t.fn();
    }
  }

  function dispatch(type) {
    (listeners[type] || []).slice().forEach((fn) => fn());
  }

  return { window, document, scrolls, runPending, dispatch, tasks };
}

test("does nothing when there is no hash", () => {
  const env = makeEnv("");
  const scroller = createHashScroller({
    window: env.window,
    document: env.document,
    attemptDelays: [0, 10],
  });

  scroller.scrollToHash();
  env.runPending();

  assert.equal(env.scrolls.length, 0);
});

test("scrolls to the hash target across every scheduled attempt", () => {
  const env = makeEnv("#docker-backend");
  const scroller = createHashScroller({
    window: env.window,
    document: env.document,
    attemptDelays: [0, 10, 20],
  });

  scroller.scrollToHash();
  env.runPending();

  assert.equal(env.scrolls.length, 3);
  assert.deepEqual(env.scrolls[0], {
    id: "docker-backend",
    options: { block: "start" },
  });
});

test("stops re-scrolling once the user scrolls", () => {
  const env = makeEnv("#docker-backend");
  const scroller = createHashScroller({
    window: env.window,
    document: env.document,
    attemptDelays: [0, 10, 20, 30],
  });

  scroller.scrollToHash();

  // Run only the first scheduled attempt.
  const first = env.tasks.find((t) => !t.cancelled);
  first.cancelled = true;
  first.fn();
  assert.equal(env.scrolls.length, 1);

  // The user scrolls; remaining attempts must be cancelled.
  env.dispatch("wheel");
  env.runPending();

  assert.equal(env.scrolls.length, 1);
});
