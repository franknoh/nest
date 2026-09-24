// Generates the site's pages from the registry: the home page reads
// `index.json` directly; each model gets a card, an architecture, and a
// files page under `models/<name>/`, with its previews copied to `public/`.
// Generated directories are ignored by git; run before `vitepress dev` or
// `vitepress build`.
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const site = resolve(here, "..");
const repo = resolve(site, "..");

function fresh(dir) {
  rmSync(dir, { recursive: true, force: true });
  mkdirSync(dir, { recursive: true });
}

// The grammar follows the Linnet checkout next to this repository when there is one.
const linnetGrammar = resolve(repo, "..", "Linnet/editors/textmate/linnet.tmLanguage.json");
if (existsSync(linnetGrammar)) {
  cpSync(linnetGrammar, join(site, ".vitepress/linnet.tmLanguage.json"));
}

const registry = JSON.parse(readFileSync(join(repo, "index.json"), "utf8"));
const pages = join(site, "models");
fresh(pages);
const previews = join(site, "public/previews");
fresh(previews);

writeFileSync(join(site, "index.md"), "---\nlayout: home\ntitle: Nest\n---\n");

function frontmatter(model, tab, title) {
  return `---\nlayout: model\nmodel: ${model.name}\ntab: ${tab}\ntitle: ${JSON.stringify(title)}\n---\n\n`;
}

function code(lang, text) {
  // A fence longer than any backtick run inside, so Markdown files with
  // their own code blocks render whole.
  const longest = Math.max(2, ...[...text.matchAll(/`+/g)].map((m) => m[0].length));
  const fence = "`".repeat(longest + 1);
  return `${fence}${lang}\n${text.endsWith("\n") ? text : text + "\n"}${fence}\n\n`;
}

for (const model of registry.models) {
  const dir = join(repo, "models", model.name);
  const out = join(pages, model.name);
  mkdirSync(out, { recursive: true });

  // ---- card: the README, then how to load it in each backend
  let card = frontmatter(model, "card", model.title);
  const readme = join(dir, "README.md");
  if (existsSync(readme)) {
    card += readFileSync(readme, "utf8").replace(/^# .*\n+/, "").trimEnd() + "\n\n";
  }
  card += "## Backends\n\nThe same source and checkpoint in each framework; `numerics` and `compile` are the loaders' options.\n\n";
  const binds = [...Object.entries(model.generics), ...Object.entries(model.check)].map(([k, v]) => `--bind ${k}=${v}`).join(" ");
  card += "### PyTorch\n\n" + code("python", `from linnet import nest\n\nmodel = nest.load("${model.name}", backend="torch", numerics="fast", compile="inductor")\nlogits = model(tokens)`);
  card += "### JAX\n\n" + code("python", `from linnet import nest\n\nf = nest.load("${model.name}", backend="jax")          # StableHLO compiled by XLA\nlogits = f(tokens)`);
  card += "### Flax NNX\n\n" + code("python", `from linnet import nest\n\nmodel = nest.load("${model.name}", backend="nnx")\nlogits = model(tokens)`);
  card += "### Command line\n\n" + code("bash", `python -m linnet.nest pull ${model.name}\nlinnet stablehlo --root ${model.root} --entry ${model.entry} ${binds} <model dir>/${model.source}`);
  writeFileSync(join(out, "index.md"), card);

  // ---- architecture: preview, entries, blocks
  let arch = frontmatter(model, "architecture", `${model.title} architecture`);
  const previewDir = join(previews, model.name);
  mkdirSync(previewDir, { recursive: true });
  const hasPreview = existsSync(join(dir, "preview.svg"));
  if (hasPreview) {
    cpSync(join(dir, "preview.svg"), join(previewDir, "preview.svg"));
    arch += `The \`${model.entry}\` entry with one level of blocks expanded. Every edge carries the tensor type the compiler inferred at that point, in the model's own generics.\n\n`;
    arch += `<img class="nest-preview nest-preview-light" src="/previews/${model.name}/preview.svg" alt="${model.title}: ${model.entry}">\n`;
    if (existsSync(join(dir, "preview-dark.svg"))) {
      cpSync(join(dir, "preview-dark.svg"), join(previewDir, "preview-dark.svg"));
      arch += `<img class="nest-preview nest-preview-dark" src="/previews/${model.name}/preview-dark.svg" alt="${model.title}: ${model.entry}">\n`;
    }
    arch += "\n";
  }
  arch += "## Entries\n\n| Entry | Signature |\n| --- | --- |\n";
  for (const entry of model.entries) {
    arch += `| \`${entry.name}\` | \`${entry.signature.replace(/^pub entry /, "")}\` |\n`;
  }
  arch += "\n## Generics\n\nThe root block's generics as this checkpoint binds them.\n\n| | |\n| --- | --- |\n";
  for (const [name, value] of Object.entries(model.generics)) {
    arch += `| \`${name}\` | \`${value}\` |\n`;
  }
  arch += "\n## Blocks\n\nEvery block of the program with its members and functions, as `linnet inspect` prints them.\n\n";
  for (const [name, text] of Object.entries(model.blocks)) {
    arch += `### \`${name}\`\n\n` + code("text", text);
  }
  writeFileSync(join(out, "architecture.md"), arch);

  // ---- files: the directory, listed like a repository and opened in place
  let files = frontmatter(model, "files", `${model.title} files`);
  const entries = model.files.filter((file) => !file.endsWith(".svg"));
  const sizes = new Map(entries.map((file) => [file, statSync(join(dir, file)).size]));
  const weightFiles = model.weights ? model.weights.files : [];

  function size(bytes) {
    if (bytes >= 1 << 20) return `${(bytes / (1 << 20)).toFixed(1)} MB`;
    if (bytes >= 1 << 10) return `${(bytes / (1 << 10)).toFixed(1)} kB`;
    return `${bytes} B`;
  }

  function language(file) {
    if (file.endsWith(".linnet")) return "linnet";
    if (file.endsWith(".toml")) return "toml";
    if (file.endsWith(".json")) return "json";
    if (file.endsWith(".md")) return "md";
    return "text";
  }

  // One glyph per kind, so a listing is readable at a glance.
  const ICONS = {
    linnet: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M4 2h5l3 3v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm4.5 1.5V5H11z" opacity=".35"/><path fill="currentColor" d="M5.5 8.5v4h4v-1h-3v-3z"/></svg>',
    json: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M4 2h5l3 3v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm4.5 1.5V5H11z" opacity=".35"/><path fill="none" stroke="currentColor" stroke-width="1.1" d="M6.6 7.6c-.8 0-.8.9-.8 1.4s0 1.4.8 1.4M9.4 7.6c.8 0 .8.9.8 1.4s0 1.4-.8 1.4"/></svg>',
    toml: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M4 2h5l3 3v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm4.5 1.5V5H11z" opacity=".35"/><path fill="currentColor" d="M5.5 8h5v1h-2v3.5h-1V9h-2z"/></svg>',
    md: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M4 2h5l3 3v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm4.5 1.5V5H11z" opacity=".35"/><path fill="currentColor" d="M5.3 12.5v-4h1l1.2 1.8 1.2-1.8h1v4h-1V10l-1.2 1.7L6.3 10v2.5z"/></svg>',
    text: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M4 2h5l3 3v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1zm4.5 1.5V5H11z" opacity=".35"/></svg>',
    folder: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M1.5 3.5A1.5 1.5 0 0 1 3 2h3l1.4 1.6H13A1.5 1.5 0 0 1 14.5 5v7A1.5 1.5 0 0 1 13 13.5H3A1.5 1.5 0 0 1 1.5 12z"/></svg>',
    weights: '<svg viewBox="0 0 16 16" width="15" height="15" aria-hidden="true"><path fill="currentColor" d="M8 1.8 14 5v6l-6 3.2L2 11V5z" opacity=".35"/><path fill="none" stroke="currentColor" stroke-width="1.1" d="M2.3 5 8 8l5.7-3M8 8v6"/></svg>',
  };

  const folders = new Map();
  for (const file of entries) {
    const cut = file.lastIndexOf("/");
    const folder = cut < 0 ? "" : file.slice(0, cut);
    if (!folders.has(folder)) folders.set(folder, []);
    folders.get(folder).push(file);
  }
  const total = [...sizes.values()].reduce((a, b) => a + b, 0);
  const tree = `https://github.com/franknoh/nest/tree/main/models/${model.name}`;

  files += `<div class="nest-files">\n`;
  files += `<div class="nest-files-head">`;
  files += `<span class="nest-files-path">${ICONS.folder}<code>models/${model.name}</code></span>`;
  files += `<span class="nest-files-meta">${entries.length + weightFiles.length} files · ${size(total)} in the registry · <a href="${tree}" target="_blank" rel="noreferrer">GitHub</a></span>`;
  files += `</div>\n\n`;

  for (const folder of [...folders.keys()].sort()) {
    if (folder) {
      files += `<div class="nest-row nest-row-folder">${ICONS.folder}<span class="nest-row-name">${folder}</span></div>\n\n`;
    }
    for (const file of folders.get(folder).sort()) {
      const name = folder ? file.slice(folder.length + 1) : file;
      const lang = language(file);
      const raw = `https://github.com/franknoh/nest/blob/main/models/${model.name}/${file}`;
      files += `<details class="nest-row nest-row-file${folder ? " nest-row-nested" : ""}">\n`;
      files += `<summary>${ICONS[lang] ?? ICONS.text}<span class="nest-row-name">${name}</span>`;
      files += `<span class="nest-row-size">${size(sizes.get(file))}</span></summary>\n\n`;
      files += `<div class="nest-row-body">\n\n`;
      files += code(lang, readFileSync(join(dir, file), "utf8"));
      files += `<a class="nest-row-raw" href="${raw}" target="_blank" rel="noreferrer">Open on GitHub</a>\n\n`;
      files += `</div>\n\n`;
      files += `</details>\n\n`;
    }
  }

  for (const file of weightFiles) {
    const href = `https://huggingface.co/${model.weights.repo}/blob/main/${file}`;
    files += `<a class="nest-row nest-row-file nest-row-remote" href="${href}" target="_blank" rel="noreferrer">`;
    files += `${ICONS.weights}<span class="nest-row-name">${file}</span>`;
    files += `<span class="nest-row-size">Hugging Face ↗</span></a>\n\n`;
  }
  files += `</div>\n`;
  writeFileSync(join(out, "files.md"), files);
}

console.log(`synced ${registry.models.length} models`);
