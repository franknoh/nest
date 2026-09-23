<script setup lang="ts">
import { useData } from "vitepress";
import { computed } from "vue";
import { byName, linkNames, parameters } from "./registry";

const { frontmatter, page } = useData();
const model = computed(() => byName(String(frontmatter.value.model)));
const tab = computed(() => String(frontmatter.value.tab ?? "card"));
const base = computed(() => `/models/${model.value?.name ?? ""}/`);
const tabs = [
  { id: "card", label: "Model card", path: "" },
  { id: "architecture", label: "Architecture", path: "architecture" },
  { id: "files", label: "Files", path: "files" },
];
void page;
</script>

<template>
  <div v-if="model" class="nest-model">
    <header class="nest-model-head">
      <div class="nest-model-title">
        <span class="nest-crumb"><a href="/">Models</a> / {{ model.name }}</span>
        <h1>{{ model.title }}</h1>
        <p>{{ model.summary }}</p>
        <div class="nest-model-badges">
          <span class="nest-badge">{{ parameters(model.parameters) }} parameters</span>
          <span v-if="model.family" class="nest-badge nest-badge-family">{{ model.family }}</span>
          <span class="nest-badge nest-badge-muted">{{ model.license }}</span>
          <span v-for="t in model.tags" :key="t" class="nest-tag">{{ t }}</span>
        </div>
      </div>
      <div class="nest-model-links">
        <a v-for="(url, key) in model.links" :key="key" :href="url" target="_blank" rel="noreferrer" class="nest-button">
          {{ linkNames[key] ?? key }}
        </a>
        <a :href="`https://github.com/franknoh/nest/tree/main/models/${model.name}`" target="_blank" rel="noreferrer" class="nest-button nest-button-muted">
          Registry entry
        </a>
      </div>
    </header>

    <nav class="nest-tabs">
      <a v-for="t in tabs" :key="t.id" :href="base + t.path" :class="{ active: tab === t.id }">{{ t.label }}</a>
    </nav>

    <div class="nest-model-body">
      <article class="nest-page vp-doc">
        <slot />
      </article>
      <aside class="nest-model-side">
        <h3>Load</h3>
        <pre class="nest-snippet"><code><span class="k">from</span> linnet <span class="k">import</span> nest

model = nest.load(<span class="s">"{{ model.name }}"</span>)</code></pre>
        <h3>Checkpoint</h3>
        <p v-if="model.weights">
          <a :href="`https://huggingface.co/${model.weights.repo}`" target="_blank" rel="noreferrer">{{ model.weights.repo }}</a>
          <br /><span class="nest-muted">{{ model.weights.files.join(", ") }}</span>
        </p>
        <h3>Entries</h3>
        <ul class="nest-side-list">
          <li v-for="e in model.entries" :key="e.name"><code>{{ e.name }}</code></li>
        </ul>
        <h3>Generics</h3>
        <ul class="nest-side-list">
          <li v-for="(value, name) in model.generics" :key="name"><code>{{ name }} = {{ value }}</code></li>
        </ul>
      </aside>
    </div>
  </div>
  <article v-else class="nest-page vp-doc">
    <slot />
  </article>
</template>
