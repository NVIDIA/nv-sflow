// Session-owned asset promises outlive a route mount. Keep successful loads
// (classic scripts cannot be evaluated twice); remove failed loads for Retry.
// CommonJS bundled by Docusaurus: avoid syntax that injects ESM Babel helpers.
const pending = new Map();

function loadOnce(url, load) {
  if (!pending.has(url)) {
    pending.set(url, Promise.resolve().then(load).catch(function (error) {
      pending.delete(url);
      throw error;
    }));
  }
  return pending.get(url);
}

function loadScript(src) {
  return loadOnce(src, function () {
    return new Promise(function (resolve, reject) {
      const el = document.createElement("script");
      el.src = src;
      el.onload = resolve;
      el.onerror = function () {
        el.remove();
        reject(new Error("Failed to load " + src));
      };
      document.body.appendChild(el);
    });
  });
}

function loadRecipeScripts(dataUrl, builderUrl) {
  const index = loadOnce(dataUrl, function () {
    return loadGzipJson(dataUrl).then(function (data) {
      window.RECIPE_DATA = data.RECIPE_DATA;
      window.MACHINE_DATA = data.MACHINE_DATA;
    });
  });
  // builder.js reads the index only in init(), after both requests finish.
  return Promise.all([index, loadScript(builderUrl)]);
}

function loadGzipJson(url) {
  return fetch(url).then(function (res) {
    if (!res.ok) throw new Error("Failed to load " + url + " (HTTP " + res.status + ")");
    return new Response(res.body.pipeThrough(new DecompressionStream("gzip"))).json();
  });
}

function loadRecipeYaml(url) {
  return loadOnce(url, function () {
    return loadGzipJson(url).then(function (data) {
      window.RECIPE_YAML = data;
    });
  });
}

module.exports = { loadRecipeScripts, loadRecipeYaml };
