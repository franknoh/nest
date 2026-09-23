<script setup lang="ts">
import { computed, ref } from "vue";
import { linkNames, models, parameters } from "./registry";

const query = ref("");
const family = ref<string | null>(null);
const tag = ref<string | null>(null);
const sort = ref<"name" | "parameters">("name");

const families = computed(() => {
  const counts = new Map<string, number>();
  for (const m of models) if (m.family) counts.set(m.family, (counts.get(m.family) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
});
const tags = computed(() => {
  const counts = new Map<string, number>();
  for (const m of models) for (const t of m.tags) counts.set(t, (counts.get(t) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1]);
});

const shown = computed(() => {
  const q = query.value.trim().toLowerCase();
  const out = models.filter((m) => {
    if (family.value && m.family !== family.value) return false;
    if (tag.value && !m.tags.includes(tag.value)) return false;
    if (!q) return true;
    const text = [m.name, m.title, m.summary, m.family ?? "", ...m.tags, m.license].join(" ").toLowerCase();
    return q.split(/\s+/).every((word) => text.includes(word));
  });
  if (sort.value === "parameters") {
    out.sort((a, b) => (b.parameters ?? 0) - (a.parameters ?? 0));
  } else {
    out.sort((a, b) => a.title.localeCompare(b.title));
  }
  return out;
});

function toggle(target: { value: string | null }, value: string) {
  target.value = target.value === value ? null : value;
}
</script>

<template>
  <div class="nest-home">
    <section class="nest-hero">
      <h1>Models that carry their architecture.</h1>
      <p>
        Every model in Nest is a checked <code>.linnet</code> source with a SafeTensors checkpoint on the
        Hugging Face Hub. Shapes and dtypes are verified before anything runs, and the same file loads in
        PyTorch, JAX, XLA, and ONNX Runtime.
      </p>
      <pre class="nest-snippet"><code><span class="k">from</span> linnet <span class="k">import</span> nest

model = nest.load(<span class="s">"tinyllama-1.1b-chat"</span>, backend=<span class="s">"torch"</span>)</code></pre>
    </section>

    <section class="nest-catalogue">
      <aside class="nest-filters">
        <label class="nest-search">
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
            <circle cx="11" cy="11" r="7" /><path d="m20 20-3.5-3.5" />
          </svg>
          <input v-model="query" type="search" placeholder="Search models" />
        </label>
        <div class="nest-filter-group">
          <h3>Family</h3>
          <button
            v-for="[name, count] in families"
            :key="name"
            type="button"
            class="nest-chip"
            :class="{ active: family === name }"
            @click="toggle(family, name)"
          >
            {{ name }} <span>{{ count }}</span>
          </button>
        </div>
        <div class="nest-filter-group">
          <h3>Tags</h3>
          <button
            v-for="[name, count] in tags"
            :key="name"
            type="button"
            class="nest-chip"
            :class="{ active: tag === name }"
            @click="toggle(tag, name)"
          >
            {{ name }} <span>{{ count }}</span>
          </button>
        </div>
      </aside>

      <div class="nest-results">
        <div class="nest-results-bar">
          <span>{{ shown.length }} {{ shown.length === 1 ? "model" : "models" }}</span>
          <label>
            Sort
            <select v-model="sort">
              <option value="name">by name</option>
              <option value="parameters">by parameters</option>
            </select>
          </label>
        </div>
        <ul class="nest-grid">
          <li v-for="m in shown" :key="m.name" class="nest-card">
            <a :href="`/models/${m.name}/`" class="nest-card-link">
              <div class="nest-card-head">
                <h2>{{ m.title }}</h2>
                <span class="nest-badge">{{ parameters(m.parameters) }}</span>
              </div>
              <p>{{ m.summary }}</p>
              <div class="nest-card-meta">
                <span v-if="m.family" class="nest-badge nest-badge-family">{{ m.family }}</span>
                <span class="nest-badge nest-badge-muted">{{ m.license }}</span>
                <span v-for="t in m.tags.slice(0, 3)" :key="t" class="nest-tag">{{ t }}</span>
              </div>
            </a>
            <div class="nest-card-links">
              <a v-for="(url, key) in m.links" :key="key" :href="url" target="_blank" rel="noreferrer">{{ linkNames[key] ?? key }}</a>
            </div>
          </li>
        </ul>
        <p v-if="shown.length === 0" class="nest-empty">No model matches. <a href="https://github.com/franknoh/nest#adding-a-model">Add one.</a></p>
      </div>
    </section>
  </div>
</template>
