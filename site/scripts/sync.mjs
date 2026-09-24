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

  // ---- files: the directory, as a tree that opens on click
  let files = frontmatter(model, "files", `${model.title} files`);
  files += `Everything in \`models/${model.name}/\`. Select a file to read it; the\ncheckpoint itself lives on the Hub.\n\n`;
  files += `<div class="nest-files">\n\n`;

  const folders = new Map();
  for (const file of model.files) {
    if (file.endsWith(".svg")) continue;
    const cut = file.lastIndexOf("/");
    const folder = cut < 0 ? "" : file.slice(0, cut);
    if (!folders.has(folder)) folders.set(folder, []);
    folders.get(folder).push(file);
  }

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

  for (const folder of [...folders.keys()].sort()) {
    if (folder) files += `<div class="nest-folder">${folder}/</div>\n\n`;
    for (const file of folders.get(folder).sort()) {
      const name = folder ? file.slice(folder.length + 1) : file;
      const bytes = statSync(join(dir, file)).size;
      files += `<details class="nest-file">\n<summary><span class="nest-file-name">${name}</span><span class="nest-file-size">${size(bytes)}</span></summary>\n\n`;
      files += code(language(file), readFileSync(join(dir, file), "utf8"));
      files += `</details>\n\n`;
    }
  }

  if (model.weights) {
    for (const file of model.weights.files) {
      const href = `https://huggingface.co/${model.weights.repo}/blob/main/${file}`;
      files += `<a class="nest-file nest-file-remote" href="${href}" target="_blank" rel="noreferrer">`;
      files += `<span class="nest-file-name">${file}</span><span class="nest-file-size">on the Hub</span></a>\n\n`;
    }
  }
  files += `</div>\n`;
  writeFileSync(join(out, "files.md"), files);
}

console.log(`synced ${registry.models.length} models`);
