import React, { useEffect, useRef, useState } from "react";
import Layout from "@theme/Layout";
import useBaseUrl from "@docusaurus/useBaseUrl";
import { loadRecipeScripts, loadRecipeYaml } from "../utils/recipeAssets";
import "./vllm-recipes.css";

// Static markup for the builder - kept as one HTML string (instead of hand-
// converted JSX) so it stays byte-for-byte in sync with the ids/classes the
// vanilla-JS logic in static/vllm-recipes/builder.js binds to via
// document.getElementById. builder.js only paints: the selection and command
// rules live in src/utils/recipeSelection.js and recipeCommand.js (tested,
// no DOM) and reach it through src/clientModules/recipeSelection.client.js.
// The data comes from vllm-recipes-sflow - see SOURCE.md.
const CONTENT_HTML = `
<div class="vr-layout">
  <aside class="vr-sidebar">
    <input id="filter" placeholder="Filter models…">
    <div id="hits" class="vr-hits"></div>
    <nav class="menu thin-scrollbar">
      <ul id="modelList" class="menu__list"></ul>
    </nav>
  </aside>

  <main class="vr-content">
    <div class="model-header">
      <h1 id="modelTitle">vLLM Recipes</h1>
      <p class="default-model" id="modelSub">Loading models…</p>
    </div>

    <div id="launcherRow"><span class="pills-label">Launcher</span><div id="launcherPills" class="pills"></div></div>
    <div id="variantRow"><span class="pills-label">Checkpoint</span><div id="variantPills" class="pills"></div></div>
    <div id="hardwareRow"><span class="pills-label">Hardware</span><div id="hardwarePills" class="pills"></div></div>
    <div id="strategyRow"><span class="pills-label">Strategy</span><div id="strategyPills" class="pills"></div></div>
    <div id="details"></div>

    <section class="card">
      <h2 id="launcherOptsTitle">Launcher settings</h2>
      <div class="opt-row" id="launcherOpts">
        <div data-launcher="slurm">
          <label class="opt" for="slurmAccount" style="cursor:default">SLURM_ACCOUNT (--set; or edit the machine profile)</label>
          <input id="slurmAccount" placeholder="your-slurm-account">
        </div>
        <div data-launcher="slurm">
          <label class="opt" for="slurmPartition" style="cursor:default">SLURM_PARTITION</label>
          <input id="slurmPartition" placeholder="your-slurm-partition">
        </div>
        <div data-launcher="docker">
          <label class="opt" for="modelHostPath" style="cursor:default">MODEL_HOST_PATH — model weights dir on the docker host (bind-mounted at the same path)</label>
          <input id="modelHostPath" placeholder="/path/to/model-weights/on-this-box">
        </div>
        <div data-launcher="k8s">
          <label class="opt" for="k8sNamespace" style="cursor:default">K8S_NAMESPACE — namespace the task pods run in</label>
          <input id="k8sNamespace" placeholder="your-k8s-namespace">
        </div>
        <div data-launcher="k8s">
          <label class="opt" for="modelPvcClaim" style="cursor:default">MODEL_PVC_CLAIM — optional: existing checkpoint PVC (mounted read-only at /models)</label>
          <input id="modelPvcClaim" placeholder="(none — download from HF each run)">
        </div>
      </div>
    </section>

    <section class="card">
      <h2>Options</h2>
      <div class="opt-row">
        <div>
          <label class="opt"><input type="checkbox" id="useLocalModel"> Use local checkpoint (skip HF download)</label>
          <input id="modelPath" placeholder="/path/to/your/local-checkpoint-dir" disabled>
        </div>
        <div>
          <label class="opt"><input type="checkbox" id="useCacheDir"> Redirect caches to shared storage (CACHE_DIR)</label>
          <input id="cacheDirPath" placeholder="/path/to/shared/cache-root" disabled>
          <div id="cacheSupport" class="note"></div>
        </div>
      </div>
      <div class="opt-row" id="ifaceRow">
        <div>
          <label class="opt" for="ifacePath" style="cursor:default">IFACE_NAME — cluster network interface for the multi-node rendezvous</label>
          <input id="ifacePath" placeholder="ib0">
        </div>
      </div>
      <div class="opt-row">
        <div>
          <label class="opt" for="mode" style="cursor:default">Run mode</label>
          <select id="mode">
            <option value="--tui">--tui (run it, interactive: submit + follow)</option>
            <option value="">(no flag — run it, submit and don't attach)</option>
            <option value="--dry-run">--dry-run (render only, don't submit)</option>
          </select>
        </div>
      </div>
    </section>

    <div id="todo" class="empty">
      <h2>Fill these in before running</h2>
      <ul id="todoList"></ul>
    </div>

    <div class="code-block">
      <div class="code-head"><span>sflow run</span><span><button id="downloadBundle" title="workflow.yaml + machine profile(s) + run.sh, flat in one directory: tar xzf, cd in, ./run.sh">Download bundle (.tar.gz)</button> <button id="downloadBundleZip" title="Same bundle as a .zip - no tar needed on Mac/Windows">Download bundle (.zip)</button> <button id="downloadSelfContained" title="workflow.yaml + machine profile(s) composed into one file (sflow compose), no -f flags needed: sflow run -f recipe.yaml">Download Self-contained Recipe</button> <button id="copy">Copy</button></span></div>
      <pre id="output"></pre>
    </div>

    <div class="code-block">
      <div class="code-head"><span id="yamlPath">workflow.yaml</span><button id="downloadYaml">Download</button></div>
      <pre id="yaml"></pre>
    </div>

    <div id="machineBlocks"></div>
  </main>
</div>
`;

export default function VllmRecipes() {
  const recipeDataUrl = useBaseUrl("/vllm-recipes/recipe_data.json.gz");
  const builderUrl = useBaseUrl("/vllm-recipes/builder.js");
  const recipeYamlGzUrl = useBaseUrl("/vllm-recipes/recipe_yaml.json.gz");
  const [error, setError] = useState("");
  const [attempt, setAttempt] = useState(0);
  const initialized = useRef(false);

  useEffect(() => {
    let active = true;
    setError("");
    // Asset requests belong to the session; only this mount owns its DOM.
    // Retrying a YAML failure reuses the initialized DOM and its listeners.
    (async () => {
      try {
        await loadRecipeScripts(recipeDataUrl, builderUrl);
        if (!active) return;
        if (!initialized.current) {
          window.vllmRecipesInit();
          initialized.current = true;
        }
        // YAML is the bulk of the payload; fetch/decompress it only after
        // the selectors and command become interactive.
        await loadRecipeYaml(recipeYamlGzUrl);
        if (!active) return;
        window.vllmRecipesOnYamlLoaded();
      } catch (err) {
        if (active) setError(err.message);
      }
    })();
    return () => { active = false; };
  }, [recipeDataUrl, builderUrl, recipeYamlGzUrl, attempt]);

  return (
    <Layout
      title="vLLM Recipes Command Builder"
      description="Pick a model + hardware already generated in vllm-recipes-sflow and get a runnable sflow run command."
    >
      {error && <div role="alert">
        Could not load recipes: {error}. <button onClick={() => setAttempt(n => n + 1)}>Retry</button>
      </div>}
      <div className="vr-page" dangerouslySetInnerHTML={{ __html: CONTENT_HTML }} />
    </Layout>
  );
}
