// Everything lives inside one function, callable more than once: this file
// is loaded once per page session (see vllm-recipes.js), but Docusaurus's
// client-side router unmounts/remounts the page's DOM on every SPA
// navigation away and back - re-querying elements and re-attaching
// listeners here is what makes that work again instead of leaving the
// sidebar bound to detached elements from the first mount.
(function () {
  // Every spot a user has to plug in their own value before this runs:
  // blank machine-profile variables, the mount-path placeholder, and the
  // generated command's own fallback placeholder text when a field is left
  // empty. Same list drives the highlight-in-place marks and the checklist.
  // `filled` (optional) drops an entry from the checklist once the matching
  // Options input covers it - the placeholder text still sits in the machine
  // profile shown below, but the command now passes the value via --set.
  const filledIn = (...ids) => () => ids.every((id) => $(id).value.trim());
  const PLACEHOLDERS = [
    { re: /REQUIRED: set for your site/g, label: "SLURM_ACCOUNT / SLURM_PARTITION — fill in above (passed as --set) or edit the machine profile below", filled: filledIn("slurmAccount", "slurmPartition") },
    { re: /REPLACE\/WITH\/YOUR\/SHARED\/FS\/ROOT/g, label: "container_mounts — shared filesystem root (machine profile)" },
    { re: /\/path\/to\/your\/local-checkpoint-dir/g, label: "MODEL — local checkpoint path (fill in above)" },
    { re: /\/path\/to\/shared\/cache-root/g, label: "CACHE_DIR — shared cache path (fill in above)" },
    { re: /\/path\/to\/model-weights\/on-this-box/g, label: "MODEL_HOST_PATH — model weights dir on the docker host (docker launcher)" },
    { re: /your-k8s-namespace/g, label: "K8S_NAMESPACE — namespace the task pods run in (k8s launcher)" },
    { re: /Cluster network interface for NCCL\/Gloo rendezvous[^"]*/g, label: "IFACE_NAME — cluster network interface for the multi-node rendezvous, e.g. ib0 (fill in above)", filled: filledIn("ifacePath") },
    // No input for this one on purpose: the command is meant to be copied
    // around, and a token pasted into it lands in shell history and logs.
    { re: /^ {2}HF_TOKEN:\n\s+value: ""/gm, label: "HF_TOKEN — this checkpoint is gated; pass --set HF_TOKEN=... yourself rather than storing it in the command" },
    // Every other "REQUIRED:" in a machine profile (K8S_NAMESPACE, and
    // MODEL_PVC_CLAIM when that fragment is composed in). Highlight only, no
    // label: the command placeholders above already list these.
    { re: /REQUIRED: (?!set for your site)[^"\n]*/g },
  ];

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlight(text) {
    let html = escapeHtml(text);
    for (const { re } of PLACEHOLDERS) {
      html = html.replace(re, (m) => `<mark class="ph">${m}</mark>`);
    }
    return html;
  }

  function renderTodo(extra, ...texts) {
    const combined = texts.join("\n");
    const labels = [
      ...PLACEHOLDERS.filter(({ re, label, filled }) => label && combined.match(re) && !(filled && filled())).map((p) => p.label),
      ...extra,
    ];
    const todo = $("todo");
    todo.classList.toggle("empty", labels.length === 0);
    $("todoList").innerHTML = labels.map((l) => `<li>${escapeHtml(l)}</li>`).join("");
  }

  // Re-bound at the top of init() every call, so every function below
  // always operates on the DOM currently attached to the page.
  let $ = (id) => document.getElementById(id);
  let filterInput, modelListEl, launcherPillsEl, variantPillsEl, hardwarePillsEl, strategyPillsEl;

  // What the user has asked for. Deliberately outside init() - preserved
  // across remounts, so navigating away and back keeps your picks instead of
  // resetting. Which of these are actually reachable, and what they settle to
  // when they are not, is decided by resolveSelection (src/utils/
  // recipeSelection.js, injected on window by the page component): this file
  // only reports clicks into `wanted` and paints what comes back.
  const wanted = {
    launcher: "slurm",
    filter: "",
    modelGroup: null,
    variant: null,
    hardware: null,
    strategy: null,
    pvcClaim: "",
  };
  let resolution = null;
  // The command as data (src/utils/recipeCommand.js), rebuilt by update();
  // the page renders it one way, the bundle's run.sh another.
  let plan = null;

  function pill(text, { selected, disabled, title }, onClick) {
    const btn = document.createElement("button");
    btn.className = "pill" + (selected ? " selected" : "") + (disabled ? " disabled" : "");
    btn.textContent = text;
    if (title) btn.title = title;
    if (disabled) btn.disabled = true;
    else btn.addEventListener("click", onClick);
    return btn;
  }

  function paint(el, options, render) {
    el.innerHTML = "";
    for (const o of options) el.appendChild(render(o));
  }

  function paintLaunchers() {
    paint(launcherPillsEl, resolution.axes.launchers, (l) =>
      pill(l.label, { selected: l.selected, title: l.title }, () => choose({ launcher: l.id })));
  }

  function paintModels() {
    modelListEl.innerHTML = "";
    const byOrg = new Map();
    for (const m of resolution.axes.models) {
      if (!byOrg.has(m.org)) byOrg.set(m.org, []);
      byOrg.get(m.org).push(m);
    }

    for (const [org, rows] of byOrg) {
      const isActiveOrg = rows.some((m) => m.selected);

      // Real Docusaurus sidebar markup (li.menu__list-item > details as the
      // collapsible > nested ul.menu__list), so this reads as the same sidebar
      // component as the docs nav: automatic indentation from the nested
      // ul's own padding-left (no manual per-level offset), the real caret
      // icon via menu__link--sublist-caret, and the active *path* highlighted
      // (category + leaf) not just the leaf, matching a real doc page's
      // current-page breadcrumb through its sidebar tree.
      const li = document.createElement("li");
      li.className = "menu__list-item";

      const group = document.createElement("details");
      // Filtering or holding the selected model opens the org by default -
      // otherwise it stays collapsed.
      group.open = !!wanted.filter.trim() || isActiveOrg;

      const summary = document.createElement("summary");
      summary.className = "menu__list-item-collapsible";
      // The caret's real Infima rule only gives it a fixed 6px left margin
      // (a *later* rule in the stylesheet overrides the auto-margin one
      // that would otherwise center-push it) - the real sidebar gets a
      // flush-right caret because its label sits in a flex:1 span that
      // eats the remaining row width, not because the caret pushes itself.
      // Bare text here would shrink-wrap instead, leaving the caret
      // sitting right next to it.
      summary.innerHTML =
        `<a class="menu__link menu__link--sublist menu__link--sublist-caret${isActiveOrg ? " menu__link--active" : ""}">` +
        `<span class="vr-org-label">${escapeHtml(org)}</span></a>`;
      group.appendChild(summary);

      const sublist = document.createElement("ul");
      sublist.className = "menu__list";
      for (const m of rows) {
        const item = document.createElement("li");
        item.className = "menu__list-item";
        const link = document.createElement("a");
        link.className = "menu__link" + (m.selected ? " menu__link--active" : "");
        link.href = "#";
        link.textContent = m.label;
        link.title = m.group;
        link.addEventListener("click", (e) => {
          e.preventDefault();
          choose({ modelGroup: m.group });
        });
        item.appendChild(link);
        sublist.appendChild(item);
      }
      group.appendChild(sublist);
      li.appendChild(group);
      modelListEl.appendChild(li);
    }
    $("hits").textContent = `${resolution.axes.models.length} / ${TOTAL_MODELS} models`;
  }

  function paintVariants() {
    // Labelled "Checkpoint", not "Precision"/"Variant" - upstream's variants
    // aren't a precision tier list, they're different served checkpoints: some
    // differ by precision (fp8/nvfp4), some by which speculative-decoding draft
    // module is bundled (dspark), some just by release date (0813).
    $("variantRow").classList.toggle("empty", resolution.axes.variants.length <= 1);
    paint(variantPillsEl, resolution.axes.variants, (v) =>
      pill(v.label, { selected: v.selected }, () => choose({ variant: v.value })));
  }

  function paintHardware() {
    paint(hardwarePillsEl, resolution.axes.hardware, (h) =>
      pill(h.label, { selected: h.selected }, () => choose({ hardware: h.id })));
  }

  function paintStrategies() {
    $("strategyRow").classList.toggle("empty", resolution.axes.strategies.length <= 1);
    paint(strategyPillsEl, resolution.axes.strategies, (s) =>
      pill(
        `${s.value} · ${s.nodes === 1 ? "1 node" : `${s.nodes} nodes`}`,
        {
          selected: s.selected,
          disabled: !!s.disabled,
          title: s.disabled ? blockedText(s.disabled) : "",
        },
        () => choose({ strategy: s.value }),
      ));
  }

  // How many models exist at all, for the "N / M models" counter - the
  // resolver reports what is reachable, this is the ceiling it is measured
  // against.
  let TOTAL_MODELS = 0;

  function selectedHardwareLabel() {
    return (resolution.axes.hardware.find((h) => h.selected) || {}).label || "";
  }

  function blockedText(block) {
    const hw = (resolution.axes.hardware.find((h) => h.selected) || {}).label || "this";
    return `${block.needGb} GB of weights don't fit one ${hw} node (${block.haveGb} GB) — use a multi-node strategy`;
  }

  // The one path that redraws anything: read the inputs the resolver needs,
  // resolve, remember what it settled on, repaint every axis, rebuild the
  // command. Every click routes through choose() so no caller has to know
  // which axes its change invalidates.
  function choose(patch) {
    Object.assign(wanted, patch);
    render();
  }

  function render() {
    wanted.filter = filterInput.value;
    wanted.pvcClaim = $("modelPvcClaim").value.trim();
    resolution = window.__vllmRecipes.resolveSelection(RECIPE_DATA, wanted);
    Object.assign(wanted, resolution.selected);
    paintLaunchers();
    paintModels();
    paintVariants();
    paintHardware();
    paintStrategies();
    update();
  }
  function update() {
    const r = resolution.recipe;
    const out = $("output");
    const details = $("details");

    if (!r) {
      plan = null;
      const n = resolution.blocked;
      $("modelTitle").textContent = n ? resolution.selected.modelGroup : "";
      $("modelSub").textContent = "";
      out.textContent = "";
      details.innerHTML = n
        ? `<div class="note">every strategy offered here needs ${n.needGb} GB and one ${selectedHardwareLabel()} node has ${n.haveGb} GB — pick other hardware, or switch to Slurm for the multi-node strategies.</div>`
        : "";
      $("yaml").textContent = ""; $("yamlPath").textContent = "workflow.yaml";
      $("machineBlocks").innerHTML = "";
      renderTodo([], "");
      return;
    }

    $("modelTitle").textContent = r.model_group + (r.variant !== "default" ? ` (${r.variant})` : "");
    $("modelSub").textContent = `default model: ${r.default_model}`;

    details.innerHTML = `<div class="note">hardware: ${selectedHardwareLabel()} · strategy: ${r.strategy}</div>`;

    // RECIPE_YAML (the ~90%-of-payload per-recipe text) loads in the
    // background after the page is already interactive - see recipe_yaml.js
    // loading in vllm-recipes.js. Show a placeholder until it lands.
    const yamlText = typeof RECIPE_YAML !== "undefined" ? RECIPE_YAML[r.path] : undefined;
    $("yamlPath").textContent = r.path;
    $("yaml").innerHTML = yamlText === undefined ? "Loading…" : highlight(yamlText);

    const launcher = wanted.launcher;
    const machineFiles = resolution.machineFiles;
    // One code block per file - these are separate files the command passes
    // separately, so they get separate blocks and separate Download buttons.
    const machineTexts = machineFiles.map((f) => MACHINE_DATA[f] || "(not committed to this repo)");
    $("machineBlocks").innerHTML = machineFiles
      .map((f, i) => `<div class="code-block"><div class="code-head"><span>machines/${escapeHtml(f)}</span>` +
        `<button data-file="${escapeHtml(f)}">Download</button></div><pre>${highlight(machineTexts[i])}</pre></div>`)
      .join("");
    const machineYamlText = machineTexts.join("\n");

    $("modelPath").disabled = !$("useLocalModel").checked;
    const supportsCache = r.has_cache_dir !== false;
    $("useCacheDir").disabled = !supportsCache;
    $("cacheDirPath").disabled = !supportsCache || !$("useCacheDir").checked;
    $("cacheSupport").textContent = supportsCache ? "" : "This recipe does not support cache redirection.";
    for (const el of $("launcherOpts").children) el.style.display = el.dataset.launcher === launcher ? "" : "none";
    // Only the multi-node/pd_cluster recipes declare it (recipe_data's
    // has_iface), and their rendezvous needs it set.
    $("ifaceRow").style.display = r.has_iface ? "" : "none";
    $("launcherOptsTitle").textContent = `${resolution.axes.launchers.find((l) => l.selected).label} settings`;
    const valueOr = (id, fallback) => $(id).value.trim() || fallback;

    // Placeholder text for the always-emitted values is supplied here, not in
    // the planner, so the checklist's placeholder scan can flag what is still
    // unfilled; the optional ones are passed as typed (blank = leave it out).
    plan = window.__vllmRecipes.planCommand(resolution, {
      launcher,
      modelHostPath: valueOr("modelHostPath", "/path/to/model-weights/on-this-box"),
      k8sNamespace: valueOr("k8sNamespace", "your-k8s-namespace"),
      pvcClaim: wanted.pvcClaim,
      // backend/k8s_model_pvc.yaml omits read_only, so sflow mounts its PVC
      // read-only. Warn before asking a cache-writing recipe to use it.
      pvcReadOnly: true,
      slurmAccount: $("slurmAccount").value.trim(),
      slurmPartition: $("slurmPartition").value.trim(),
      localModel: $("useLocalModel").checked ? valueOr("modelPath", "/path/to/your/local-checkpoint-dir") : null,
      ifaceName: $("ifacePath").value.trim(),
      cacheDir: $("useCacheDir").checked ? valueOr("cacheDirPath", "/path/to/shared/cache-root") : null,
      mode: $("mode").value,
      recipeYaml: yamlText,
    });
    const commandText = window.__vllmRecipes.renderCommand(plan);
    out.innerHTML = highlight(commandText);
    // The planner says what will bite; this decides where and how to say it.
    // Recipe-level problems go under the title, fill-in problems join the
    // checklist next to the other placeholders.
    const extraTodos = [];
    for (const w of plan.warnings) {
      if (w.kind === "no_docker_image") {
        details.innerHTML += `<div class="note">this recipe predates the DOCKER_IMAGE variable (it still sets container_image per task) and fails on docker/k8s with "operator override key(s) not valid" — regenerate it with vllm-recipes-sflow's current generate.py first.</div>`;
      } else if (w.kind === "cache_unmounted") {
        extraTodos.push(`CACHE_DIR (${w.cacheDir}) has nowhere to persist — this pod mounts no volume; name a PVC above to keep the download`);
      } else if (w.kind === "cache_outside_mount") {
        extraTodos.push(`CACHE_DIR (${w.cacheDir}) sits outside ${w.mountName} (${w.mount}) — only that path survives the container, so the checkpoint downloads again on every run`);
      } else if (w.kind === "cache_read_only") {
        extraTodos.push(`CACHE_DIR (${w.cacheDir}) is inside a read-only PVC. Before running, set read_only: false on the models volume in backend/k8s_model_pvc.yaml (k8s_model_pvc.yaml in a bundle, or backends.compute.volumes in a self-contained recipe).`);
      }
    }

    renderTodo(extraTodos, commandText, yamlText || "", machineYamlText);
  }

  function downloadBlob(filename, blob) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  function downloadText(filename, text) {
    downloadBlob(filename, new Blob([text], { type: "text/yaml" }));
  }

  // Minimal ustar writer: enough for a handful of small text files, no
  // dependency. Gzipped with the browser's native CompressionStream (the same
  // baseline recipe_yaml.json.gz already relies on), so the result is a plain
  // `tar xzf` on the Linux box the command is meant to run on.
  function tarGz(files) {
    const enc = new TextEncoder();
    const blocks = [];
    const octal = (n, len) => n.toString(8).padStart(len - 1, "0") + "\0";
    const put = (buf, off, str) => { for (let i = 0; i < str.length; i++) buf[off + i] = str.charCodeAt(i); };
    for (const { name, text, mode = 0o644 } of files) {
      const data = enc.encode(text);
      // ustar splits long paths into prefix/name at a slash (name <= 100 bytes).
      let prefix = "", base = name;
      if (base.length > 100) {
        const cut = base.lastIndexOf("/", 155);
        prefix = base.slice(0, cut);
        base = base.slice(cut + 1);
      }
      const h = new Uint8Array(512);
      put(h, 0, base);
      put(h, 100, octal(mode, 8));
      put(h, 108, octal(0, 8));
      put(h, 116, octal(0, 8));
      put(h, 124, octal(data.length, 12));
      put(h, 136, octal(Math.floor(Date.now() / 1000), 12));
      put(h, 148, "        ");
      put(h, 156, "0");
      put(h, 257, "ustar\0" + "00");
      put(h, 345, prefix);
      let sum = 0;
      for (const b of h) sum += b;
      put(h, 148, sum.toString(8).padStart(6, "0") + "\0 ");
      blocks.push(h, data, new Uint8Array((512 - (data.length % 512)) % 512));
    }
    blocks.push(new Uint8Array(1024));
    return new Response(new Blob(blocks).stream().pipeThrough(new CompressionStream("gzip"))).blob();
  }

  // Minimal store-only (uncompressed) ZIP writer - a .tar.gz needs `tar` on
  // the PATH, which Windows only grew recently and inconsistently; a .zip
  // opens natively everywhere (Explorer, Finder, Archive Utility), so this is
  // the "just works" option for whoever isn't running the command on Linux.
  // No compression: these are a handful of small text files, and skipping
  // deflate keeps this to one CRC32 pass with no dependency.
  function crc32(bytes) {
    let c = ~0;
    for (let i = 0; i < bytes.length; i++) {
      c ^= bytes[i];
      for (let k = 0; k < 8; k++) c = (c >>> 1) ^ (0xedb88320 & -(c & 1));
    }
    return ~c >>> 0;
  }

  function zipStore(files) {
    const enc = new TextEncoder();
    const dosTime = 0, dosDate = 0x21; // arbitrary fixed timestamp, nobody reads it
    const localParts = [];
    const centralParts = [];
    let offset = 0;

    files.forEach(({ name, text }) => {
      const nameBytes = enc.encode(name);
      const data = enc.encode(text);
      const crc = crc32(data);

      const local = new DataView(new ArrayBuffer(30));
      local.setUint32(0, 0x04034b50, true);
      local.setUint16(4, 20, true); // version needed
      local.setUint16(6, 0, true); // flags
      local.setUint16(8, 0, true); // method: store
      local.setUint16(10, dosTime, true);
      local.setUint16(12, dosDate, true);
      local.setUint32(14, crc, true);
      local.setUint32(18, data.length, true);
      local.setUint32(22, data.length, true);
      local.setUint16(26, nameBytes.length, true);
      local.setUint16(28, 0, true); // extra field length
      localParts.push(new Uint8Array(local.buffer), nameBytes, data);

      const central = new DataView(new ArrayBuffer(46));
      central.setUint32(0, 0x02014b50, true);
      central.setUint16(4, 20, true); // version made by
      central.setUint16(6, 20, true); // version needed
      central.setUint16(8, 0, true); // flags
      central.setUint16(10, 0, true); // method: store
      central.setUint16(12, dosTime, true);
      central.setUint16(14, dosDate, true);
      central.setUint32(16, crc, true);
      central.setUint32(20, data.length, true);
      central.setUint32(24, data.length, true);
      central.setUint16(28, nameBytes.length, true);
      central.setUint16(30, 0, true); // extra field length
      central.setUint16(32, 0, true); // comment length
      central.setUint16(34, 0, true); // disk number
      central.setUint16(36, 0, true); // internal attrs
      central.setUint32(38, 0, true); // external attrs
      central.setUint32(42, offset, true); // local header offset
      centralParts.push(new Uint8Array(central.buffer), nameBytes);

      offset += local.buffer.byteLength + nameBytes.length + data.length;
    });

    const centralSize = centralParts.reduce((n, p) => n + p.length, 0);
    const end = new DataView(new ArrayBuffer(22));
    end.setUint32(0, 0x06054b50, true);
    end.setUint16(4, 0, true); // disk number
    end.setUint16(6, 0, true); // disk with central dir
    end.setUint16(8, files.length, true);
    end.setUint16(10, files.length, true);
    end.setUint32(12, centralSize, true);
    end.setUint32(16, offset, true); // central dir offset
    end.setUint16(20, 0, true); // comment length

    return new Blob(localParts.concat(centralParts, [new Uint8Array(end.buffer)]));
  }

  function init() {
    // Metadata and this script download concurrently; the page calls init()
    // only once both have loaded, so there are no eager reads of RECIPE_DATA.
    TOTAL_MODELS = new Set(RECIPE_DATA.map((r) => r.model_group)).size;
    filterInput = $("filter");
    modelListEl = $("modelList");
    launcherPillsEl = $("launcherPills");
    variantPillsEl = $("variantPills");
    hardwarePillsEl = $("hardwarePills");
    strategyPillsEl = $("strategyPills");

    // The PVC claim changes which machine files are selected, so it goes
    // through the resolver like the filter does; the rest only affect --set.
    filterInput.addEventListener("input", render);
    $("modelPvcClaim").addEventListener("input", render);
    for (const id of ["useLocalModel", "modelPath", "useCacheDir", "cacheDirPath", "mode",
                      "slurmAccount", "slurmPartition", "modelHostPath", "k8sNamespace", "ifacePath"]) {
      $(id).addEventListener("input", update);
      $(id).addEventListener("change", update);
    }
    $("copy").addEventListener("click", () => navigator.clipboard.writeText($("output").textContent));
    $("downloadYaml").addEventListener("click", () => {
      const r = resolution.recipe;
      const yamlText = r && typeof RECIPE_YAML !== "undefined" ? RECIPE_YAML[r.path] : undefined;
      if (yamlText !== undefined) downloadText(`${r.hf_id.replace(/\//g, "_")}-${r.hardware}-workflow.yaml`, yamlText);
    });
    // Delegated: the blocks are rebuilt on every update().
    $("machineBlocks").addEventListener("click", (e) => {
      const f = e.target.dataset && e.target.dataset.file;
      if (f) downloadText(f.split("/").pop(), MACHINE_DATA[f] || "");
    });
    // Shared by both bundle formats: workflow.yaml + machine profile(s) +
    // run.sh, flat in one wrapping directory named like the archive (so
    // extracting either one leaves a single directory, not four loose files).
    function bundleFiles() {
      const r = resolution.recipe;
      const yamlText = r && typeof RECIPE_YAML !== "undefined" ? RECIPE_YAML[r.path] : undefined;
      if (yamlText === undefined) return null;
      const launcher = wanted.launcher;
      const stem = `${r.hf_id.replace(/\//g, "_")}-${r.hardware}-${launcher}`;
      const { renderCommand, basename } = window.__vllmRecipes;

      // The page's command carries repo-relative paths because it assumes a
      // vllm-recipes-sflow checkout; the bundle is for people who have none,
      // so the same plan is rendered flat and the files sit beside run.sh.
      const command = renderCommand(plan, { flat: true });

      const files = [
        { name: `${stem}/workflow.yaml`, text: yamlText },
        ...resolution.machineFiles.map((f) => ({ name: `${stem}/${basename(f)}`, text: MACHINE_DATA[f] || "" })),
        {
          name: `${stem}/run.sh`,
          mode: 0o755,
          text: `#!/usr/bin/env bash\n# ${r.model_group} on ${selectedHardwareLabel()} via ${launcher} - generated by the vLLM Recipes command builder.\n# Fill in the placeholders below (see the checklist on the page) before running.\nset -euo pipefail\ncd "$(dirname "$0")"\n${command}\n`,
        },
      ];
      return { stem, files };
    }

    $("downloadBundle").addEventListener("click", async () => {
      const b = bundleFiles();
      if (!b) return;
      downloadBlob(`${b.stem}.tar.gz`, await tarGz(b.files));
    });

    $("downloadBundleZip").addEventListener("click", () => {
      const b = bundleFiles();
      if (!b) return;
      downloadBlob(`${b.stem}.zip`, zipStore(b.files));
    });

    $("downloadSelfContained").addEventListener("click", () => {
      const r = resolution.recipe;
      const yamlText = r && typeof RECIPE_YAML !== "undefined" ? RECIPE_YAML[r.path] : undefined;
      if (yamlText === undefined) return;
      const { composeYaml } = window.__vllmRecipes;
      const fragments = [{ label: r.path, text: yamlText }].concat(
        resolution.machineFiles.map((f) => ({ label: `machines/${f}`, text: MACHINE_DATA[f] || "" })),
      );
      let composed;
      try {
        composed = composeYaml(fragments, plan.sets);
      } catch (err) {
        alert(`Could not compose this recipe: ${err.message}`);
        return;
      }
      const stem = `${r.hf_id.replace(/\//g, "_")}-${r.hardware}-${wanted.launcher}`;
      const filename = `${stem}.self-contained.yaml`;
      // No -f/--set flags needed - the overrides are already baked into the
      // file's variables - so unlike the bundle's run.sh this is the whole
      // command; show it as a header since there is nowhere else on the page
      // that names it.
      const runCommand = `sflow run -f ${filename}${plan.mode ? " " + plan.mode : ""}`;
      const header = `# ${runCommand}\n# Fill in the placeholders below (see the checklist on the page) before running.\n\n`;
      downloadText(filename, header + composed);
    });

    render();
  }

  window.vllmRecipesInit = init;
  // Called once recipe_yaml.js (loaded lazily, in the background - see
  // vllm-recipes.js) resolves, so whichever recipe is showing swaps its
  // "Loading…" placeholder for the real yaml text.
  window.vllmRecipesOnYamlLoaded = update;
})();
