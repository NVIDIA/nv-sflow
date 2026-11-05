// Composes a recipe's workflow.yaml with its machine fragments into one
// self-contained YAML file - the same result `sflow compose -f a -f b -f c`
// would produce, since sflow itself merges `-f` files this way at run time.
// There is no server here (this page is a static site), so the merge is
// reimplemented in JS against the same fragments the page already ships
// (RECIPE_YAML / MACHINE_DATA) rather than shelling out.
//
// Mirrors sflow.config.loader.merge_config_dicts (src/sflow/config/loader.py):
// version must agree across files; variables/artifacts/backends/operators are
// merged by `name` (same-name entries deep-merge so a definition can be
// scattered across files, e.g. machines/backend/k8s_model_pvc.yaml adding
// `volumes` to the `compute` backend machines/backend/k8s.yaml already
// defines); workflow.tasks merges by name the same way; every other value
// (scalars, lists) has the later file win outright - that's how
// machines/backend/k8s.yaml's list-shaped `probes.readiness` replaces the
// recipe's dict-shaped one instead of merging into it.
//
// SYNTAX CONSTRAINT - CommonJS bundled by Docusaurus's webpack: no for...of,
// no object/array spread, no classes, no async. See recipeSelection.js.

const yaml = require("js-yaml");

const NAMED_SECTION_KEYS = ["variables", "artifacts", "backends", "operators", "storage"];

function isPlainObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

// A named section may already be name-keyed ({NAME: {...}}) or a list of
// dicts with a `name` field ([{name: NAME, ...}]) - recipe/machine fragments
// only ever use the list form, but sflow accepts both, so this stays faithful
// to _section_to_merge_dict.
function sectionToMergeDict(section) {
  if (section == null) return {};
  if (isPlainObject(section)) return Object.assign({}, section);
  if (Array.isArray(section)) {
    const result = {};
    section.forEach(function (item) {
      if (isPlainObject(item) && item.name != null) {
        const entry = {};
        Object.keys(item).forEach(function (k) {
          if (k !== "name") entry[k] = item[k];
        });
        result[item.name] = entry;
      }
    });
    return result;
  }
  return {};
}

function tasksToList(tasks) {
  if (tasks == null) return [];
  if (Array.isArray(tasks)) return tasks;
  if (isPlainObject(tasks)) {
    return Object.keys(tasks).map(function (k) {
      const v = tasks[k];
      return isPlainObject(v) ? Object.assign({}, v, { name: k }) : { name: k };
    });
  }
  return [];
}

// Dicts merge key-by-key; anything else (scalars, lists, or a dict-vs-scalar
// mismatch) is a leaf where the incoming value wins outright.
function deepMerge(base, incoming) {
  if (isPlainObject(base) && isPlainObject(incoming)) {
    const result = Object.assign({}, base);
    Object.keys(incoming).forEach(function (k) {
      result[k] = k in result ? deepMerge(result[k], incoming[k]) : incoming[k];
    });
    return result;
  }
  return incoming;
}

function mergeNamedSection(baseSection, incomingSection) {
  const merged = sectionToMergeDict(baseSection);
  const incoming = sectionToMergeDict(incomingSection);
  Object.keys(incoming).forEach(function (name) {
    merged[name] = name in merged ? deepMerge(merged[name], incoming[name]) : incoming[name];
  });
  return merged;
}

// Merges workflow.tasks by name, preserving first-seen order; unnamed tasks
// (schema-invalid, but not this module's job to reject) are kept, appended.
function mergeTasks(baseTasks, incomingTasks) {
  const merged = {};
  const order = [];
  const unnamed = [];

  function absorb(list) {
    list.forEach(function (t) {
      const name = isPlainObject(t) ? t.name : null;
      if (name == null) {
        unnamed.push(t);
        return;
      }
      if (name in merged) {
        merged[name] = deepMerge(merged[name], t);
      } else {
        merged[name] = t;
        order.push(name);
      }
    });
  }

  absorb(tasksToList(baseTasks));
  absorb(tasksToList(incomingTasks));
  return order.map(function (n) { return merged[n]; }).concat(unnamed);
}

function mergeWorkflow(baseWf, incWf) {
  const result = Object.assign({}, baseWf);

  if (incWf.name != null) result.name = incWf.name;

  if (incWf.variables != null) {
    result.variables = mergeNamedSection(result.variables, incWf.variables);
  }

  if (incWf.tasks != null) {
    result.tasks = mergeTasks(result.tasks, incWf.tasks);
  }

  Object.keys(incWf).forEach(function (k) {
    if (k === "name" || k === "variables" || k === "tasks" || incWf[k] == null) return;
    result[k] = k in result ? deepMerge(result[k], incWf[k]) : incWf[k];
  });

  return result;
}

// Drops keys with a null/undefined value, recursively, so the composed YAML
// doesn't carry `field: null` for every optional key a fragment left unset.
function stripNullValues(value) {
  if (Array.isArray(value)) return value.map(stripNullValues);
  if (isPlainObject(value)) {
    const result = {};
    Object.keys(value).forEach(function (k) {
      if (value[k] != null) result[k] = stripNullValues(value[k]);
    });
    return result;
  }
  return value;
}

// A name-keyed section merges back to a list, the shape the recipe/machine
// fragments and sflow's schema both use.
function namedSectionToList(section) {
  return Object.keys(section).map(function (name) {
    return Object.assign({ name: name }, section[name]);
  });
}

// fragments: [{label, text}], in the same order `sflow run -f ...` would
// pass them (recipe workflow.yaml first, then each machine fragment).
// sets: the command plan's literal [name, value] overrides, applied last.
// Returns the composed YAML text.
function composeYaml(fragments, sets = []) {
  let merged = {};
  fragments.forEach(function (fragment) {
    const data = yaml.load(fragment.text);
    if (!isPlainObject(data)) {
      throw new Error(fragment.label + " is not a YAML mapping");
    }

    if (data.version != null) {
      if (merged.version != null && merged.version !== data.version) {
        throw new Error(
          "Version conflict: '" + merged.version + "' vs '" + data.version +
          "' (from " + fragment.label + ")",
        );
      }
      merged.version = data.version;
    }

    NAMED_SECTION_KEYS.forEach(function (key) {
      if (data[key] != null) {
        merged[key] = mergeNamedSection(merged[key], data[key]);
      }
    });

    if (data.workflow != null) {
      merged.workflow = mergeWorkflow(merged.workflow || {}, data.workflow);
    }
  });

  sets.forEach(function (kv) {
    const name = kv[0];
    if (!merged.variables || !Object.prototype.hasOwnProperty.call(merged.variables, name)) {
      throw new Error("Variable '" + name + "' is not defined in the configuration");
    }
    merged.variables[name] = Object.assign({}, merged.variables[name], { value: kv[1] });
  });

  NAMED_SECTION_KEYS.forEach(function (key) {
    if (merged[key]) merged[key] = namedSectionToList(merged[key]);
  });
  if (merged.workflow && merged.workflow.variables) {
    merged.workflow.variables = namedSectionToList(merged.workflow.variables);
  }

  const cleaned = stripNullValues(merged);
  return yaml.dump(cleaned, { sortKeys: false, lineWidth: -1, noRefs: true });
}

module.exports = { composeYaml, deepMerge, mergeNamedSection, mergeTasks, mergeWorkflow };
