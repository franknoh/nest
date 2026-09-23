import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vitepress";

const here = dirname(fileURLToPath(import.meta.url));

// The same TextMate grammar the Linnet editors use, so sources highlight
// exactly like VS Code does. `scripts/sync.mjs` refreshes the copy from a
// Linnet checkout when one is next to this repository.
const grammar = JSON.parse(readFileSync(resolve(here, "linnet.tmLanguage.json"), "utf8"));
grammar.name = "linnet";

export default defineConfig({
  title: "Nest",
  titleTemplate: ":title · Nest",
  description: "Nest, the Linnet model zoo: checked model sources with SafeTensors checkpoints, runnable in PyTorch, JAX, XLA, and ONNX Runtime.",
  lang: "en-US",
  cleanUrls: true,
  appearance: "dark",
  sitemap: { hostname: "https://nest.franknoh.dev" },
  head: [
    ["link", { rel: "icon", type: "image/svg+xml", href: "/logo.svg" }],
    ["meta", { name: "theme-color", content: "#050505" }],
  ],
  markdown: {
    languages: [grammar],
    theme: { light: "github-light", dark: "github-dark-default" },
  },
  themeConfig: {},
});
