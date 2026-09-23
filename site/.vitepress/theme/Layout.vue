<script setup lang="ts">
import { Content, useData } from "vitepress";
import { computed } from "vue";
import Home from "./Home.vue";
import ModelPage from "./ModelPage.vue";

const { frontmatter, isDark } = useData();
const layout = computed(() => frontmatter.value.layout ?? "page");

function toggle() {
  isDark.value = !isDark.value;
}
</script>

<template>
  <div class="nest">
    <header class="nest-nav">
      <div class="nest-nav-inner">
        <a class="nest-brand" href="/">
          <img src="/logo.svg" alt="" width="22" height="22" />
          <span>Nest</span>
          <span class="nest-brand-sub">models for Linnet</span>
        </a>
        <nav class="nest-links">
          <a href="/">Models</a>
          <a href="https://github.com/franknoh/nest#adding-a-model" target="_blank" rel="noreferrer">Add a model</a>
          <a href="https://linnet.franknoh.dev" target="_blank" rel="noreferrer">Linnet</a>
          <a class="nest-icon" href="https://github.com/franknoh/nest" target="_blank" rel="noreferrer" aria-label="GitHub">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="currentColor" aria-hidden="true">
              <path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 9.39 7.86 10.91.58.1.79-.25.79-.56v-2.17c-3.2.7-3.87-1.37-3.87-1.37-.52-1.33-1.28-1.68-1.28-1.68-1.04-.71.08-.7.08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.69 1.25 3.35.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.68 0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.78 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.81 1.19 1.83 1.19 3.09 0 4.41-2.69 5.38-5.26 5.67.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5z" />
            </svg>
          </a>
          <button class="nest-icon" type="button" @click="toggle" :aria-label="isDark ? 'Light mode' : 'Dark mode'">
            <svg v-if="isDark" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
            </svg>
            <svg v-else viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
              <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
            </svg>
          </button>
        </nav>
      </div>
    </header>

    <main class="nest-main">
      <Home v-if="layout === 'home'" />
      <ModelPage v-else-if="layout === 'model'">
        <Content />
      </ModelPage>
      <article v-else class="nest-page vp-doc">
        <Content />
      </article>
    </main>

    <footer class="nest-footer">
      <div class="nest-footer-inner">
        <span>Nest is the model zoo for <a href="https://linnet.franknoh.dev">Linnet</a>. Cards and sources are MIT; each checkpoint keeps its own license.</span>
        <a href="https://github.com/franknoh/nest">github.com/franknoh/nest</a>
      </div>
    </footer>
  </div>
</template>
