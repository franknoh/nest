// The registry's `index.json` (aliased in `config.ts`, so a scratch copy can stand in).
import registry from "@nest/index.json";

export interface Entry {
  name: string;
  signature: string;
  inputs: { name: string; type: string }[];
  results: string[];
  states: string[];
}

export interface Model {
  name: string;
  title: string;
  summary: string;
  license: string;
  family: string | null;
  tags: string[];
  links: Record<string, string>;
  source: string;
  root: string;
  module: string;
  entry: string;
  generics: Record<string, number | string>;
  check: Record<string, number | string>;
  parameters: number | null;
  entries: Entry[];
  blocks: Record<string, string>;
  weights: { repo: string; files: string[]; revision: string | null; bindings: string | null } | null;
  files: string[];
}

export const models: Model[] = (registry as { models: Model[] }).models;

export function byName(name: string): Model | undefined {
  return models.find((m) => m.name === name);
}

export function parameters(count: number | null): string {
  if (count == null) return "?";
  for (const [unit, size] of [["B", 1e9], ["M", 1e6], ["K", 1e3]] as const) {
    if (count >= size) return `${(count / size).toFixed(1).replace(/\.0$/, "")}${unit}`;
  }
  return String(count);
}

export const linkNames: Record<string, string> = {
  huggingface: "Hugging Face",
  github: "GitHub",
  arxiv: "arXiv",
  homepage: "Homepage",
};
