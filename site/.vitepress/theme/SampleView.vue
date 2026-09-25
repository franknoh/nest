<script setup lang="ts">
// The samples tab: `models/<name>/samples.json`, written by each benchmark
// family's `sample()` in `bench/methods/`. One small block per `kind`, with
// the reference stack's answer beside Linnet's wherever the sample has one.
// `scripts/sync.mjs` has already rewritten image paths to served addresses.
import { computed } from "vue";

type Sample = Record<string, any>;

const props = defineProps<{ data: Sample | null; title: string }>();
const s = computed(() => props.data ?? null);
const kind = computed(() => String(s.value?.kind ?? ""));

function pct(p: number | null | undefined): string {
  if (typeof p !== "number") return "–";
  const v = p * 100;
  if (v >= 10) return `${v.toFixed(1)}%`;
  if (v >= 0.1) return `${v.toFixed(2)}%`;
  return v === 0 ? "0%" : `${v.toExponential(1)}%`;
}
function fixed(v: unknown, digits = 3): string {
  return typeof v === "number" && Number.isFinite(v) ? v.toFixed(digits) : "–";
}

// ---- text generation

const agreement = computed(() => {
  const x = s.value;
  if (!x || kind.value !== "text") return "";
  const tokens = Number(x.tokens ?? 0);
  const agree = Number(x.agreeing_prefix ?? 0);
  if (!tokens) return "";
  if (agree >= tokens) return `All ${tokens} generated tokens agree with the reference.`;
  return `The first ${agree} of ${tokens} generated tokens agree with the reference; after that the two stacks round a near-tie differently and part ways, which bf16 greedy decoding allows.`;
});

// ---- classification: Linnet's top labels with the reference's probability beside each

const classes = computed(() => {
  const x = s.value;
  if (!x || kind.value !== "classification") return [];
  const reference = new Map<string, number>((x.reference ?? []).map((r: any) => [r.label, r.p]));
  return (x.top ?? []).map((t: any) => ({ label: t.label, p: t.p, ref: reference.get(t.label) }));
});
const sameTop = computed(() => {
  const x = s.value;
  if (!x?.reference?.length || !x?.top?.length) return null;
  return x.top.map((t: any) => t.label).join("\u0000") === x.reference.map((t: any) => t.label).join("\u0000");
});

// ---- similarity: sentence x sentence (text encoders) or image x label (SigLIP)

const square = computed(() => {
  const x = s.value;
  if (!x || kind.value !== "similarity") return false;
  const m = x.matrix ?? [];
  return !x.images?.length && m.length === (x.sentences ?? []).length && m.every((r: any[]) => r.length === m.length);
});
const referenceMatrix = computed<number[][] | null>(() => s.value?.reference_matrix ?? (Array.isArray(s.value?.reference) ? s.value!.reference : null));
const matrixGap = computed(() => {
  const ours: number[][] = s.value?.matrix ?? [];
  const theirs = referenceMatrix.value;
  if (!theirs) return null;
  let gap = 0;
  ours.forEach((row, i) => row.forEach((v, j) => {
    const t = theirs[i]?.[j];
    if (typeof t === "number") gap = Math.max(gap, Math.abs(v - t));
  }));
  return gap;
});
// One hue, light to dark, from zero to the largest value off the diagonal
// (a sentence against itself is always 1 and would wash the rest out); the
// diagonal takes the darkest step. Below zero is as unrelated as zero for
// reading the matrix, and every cell prints its number anyway.
const top = computed(() => {
  const m: number[][] = s.value?.matrix ?? [];
  let best = 0;
  m.forEach((row, i) => row.forEach((v, j) => {
    if ((!square.value || i !== j) && typeof v === "number") best = Math.max(best, v);
  }));
  return best > 0 ? Math.min(1, best) : 1;
});
function level(v: number): number {
  return Math.round(Math.max(0, Math.min(1, v / top.value)) * 88);
}
function shade(v: number): Record<string, string> {
  return { background: `color-mix(in srgb, var(--nest-bench-best) ${level(v)}%, var(--vp-c-bg))` };
}
// Past half-way the cell is dark enough to need light ink.
function deep(v: number): boolean {
  return level(v) > 62;
}
function gapText(g: number): string {
  if (g === 0) return "0";
  return g < 1e-3 ? g.toExponential(1) : g.toFixed(4);
}

// ---- transcription

const sameText = computed(() => {
  const x = s.value;
  if (!x || kind.value !== "transcription") return null;
  if (x.reference_text == null) return null;
  return String(x.text).trim() === String(x.reference_text).trim();
});

const PSNR: Record<string, string> = {
  linnet_vs_input: "Linnet vs the input",
  diffusers_vs_input: "diffusers vs the input",
  linnet_vs_diffusers: "Linnet vs diffusers",
};
function psnrRows(x: Sample): [string, string][] {
  return Object.entries(x.psnr_db ?? {}).map(([k, v]) => [PSNR[k] ?? k.replace(/_/g, " "), `${fixed(v, 2)} dB`]);
}
</script>

<template>
  <div class="nest-sample">
    <p v-if="!s" class="nest-empty-state">
      No sample has been recorded for {{ title }} yet. What it produces, next to the reference stack's answer, will appear here once the benchmark harness has run it.
    </p>

    <!-- text generation -->
    <template v-else-if="kind === 'text'">
      <p class="nest-sample-lede">Greedy decoding of one {{ s.chat ? "chat prompt" : "continuation" }}, through Linnet and through {{ s.reference_stack ?? "the reference stack" }}.</p>
      <div class="nest-sample-prompt">
        <span class="nest-sample-label">{{ s.chat ? "Prompt" : "Text to continue" }}</span>
        <p>{{ s.prompt }}</p>
      </div>
      <div class="nest-sample-pair">
        <figure class="nest-sample-text">
          <figcaption><i class="nest-swatch is-linnet"></i>Linnet</figcaption>
          <p>{{ s.output }}</p>
        </figure>
        <figure v-if="s.reference != null" class="nest-sample-text">
          <figcaption><i class="nest-swatch is-reference"></i>{{ s.reference_stack ?? "Reference" }}</figcaption>
          <p>{{ s.reference }}</p>
        </figure>
      </div>
      <p v-if="agreement" class="nest-sample-note">{{ agreement }}</p>
    </template>

    <!-- classification -->
    <template v-else-if="kind === 'classification'">
      <p class="nest-sample-lede">The top {{ classes.length }} classes for one photo, through Linnet and through {{ s.reference_stack ?? "the reference stack" }}.</p>
      <div class="nest-sample-split">
        <img v-if="s.image" class="nest-sample-img" :src="s.image" alt="The input photo" loading="lazy" />
        <table class="nest-sample-probs">
          <thead>
            <tr><th>Class</th><th class="num">Linnet</th><th v-if="s.reference" class="num">Reference</th></tr>
          </thead>
          <tbody>
            <tr v-for="c in classes" :key="c.label">
              <td>
                <span class="nest-sample-class">{{ c.label }}</span>
                <span class="nest-sample-meter" aria-hidden="true"><span :style="{ width: `${Math.max(0, Math.min(1, c.p)) * 100}%` }"></span></span>
              </td>
              <td class="num">{{ pct(c.p) }}</td>
              <td v-if="s.reference" class="num">{{ pct(c.ref) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="sameTop != null" class="nest-sample-note">
        {{ sameTop ? "Both stacks rank the same classes in the same order." : "The two stacks rank these classes differently." }}
      </p>
    </template>

    <!-- similarity: sentence x sentence -->
    <template v-else-if="kind === 'similarity' && square">
      <p class="nest-sample-lede">Cosine similarity between {{ s.sentences.length }} sentences{{ s.caption ? ". " : "." }}{{ s.caption?.replace(/`/g, "") }}</p>
      <ol class="nest-sample-sentences">
        <li v-for="(sentence, i) in s.sentences" :key="i">{{ sentence }}</li>
      </ol>
      <div class="nest-sample-matrix-wrap">
        <table class="nest-sample-matrix">
          <thead>
            <tr><th></th><th v-for="(_, j) in s.sentences" :key="j" scope="col">{{ j + 1 }}</th></tr>
          </thead>
          <tbody>
            <tr v-for="(row, i) in s.matrix" :key="i">
              <th scope="row">{{ i + 1 }}</th>
              <td v-for="(v, j) in row" :key="j" :style="shade(v)" :class="{ 'is-deep': deep(v) }">{{ fixed(v, 2) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p class="nest-sample-note">
        Shading runs from 0 to {{ top.toFixed(2) }}, the closest pair of different sentences.
        <template v-if="matrixGap != null">Largest difference from {{ s.reference_stack ?? "the reference" }}'s matrix: {{ gapText(matrixGap) }}.</template>
      </p>
    </template>

    <!-- similarity: image x labels -->
    <template v-else-if="kind === 'similarity'">
      <p class="nest-sample-lede">How well each caption matches the photo (sigmoid of the image-text logit), through Linnet and through {{ s.reference_stack ?? "the reference stack" }}.</p>
      <div class="nest-sample-split">
        <img v-for="(src, k) in s.images ?? []" :key="k" class="nest-sample-img" :src="src" alt="The input photo" loading="lazy" />
        <table class="nest-sample-probs">
          <thead>
            <tr><th>Caption</th><th class="num">Linnet</th><th v-if="referenceMatrix" class="num">Reference</th></tr>
          </thead>
          <tbody>
            <tr v-for="(sentence, i) in s.sentences" :key="i">
              <td>{{ sentence }}</td>
              <td class="num nest-sample-cell" :style="shade(s.matrix?.[0]?.[i] ?? 0)" :class="{ 'is-deep': deep(s.matrix?.[0]?.[i] ?? 0) }">{{ pct(s.matrix?.[0]?.[i]) }}</td>
              <td v-if="referenceMatrix" class="num">{{ pct(referenceMatrix[0]?.[i]) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p v-if="matrixGap != null" class="nest-sample-note">Largest difference from the reference: {{ gapText(matrixGap) }}.</p>
    </template>

    <!-- an image in, an image out -->
    <template v-else-if="kind === 'image'">
      <div class="nest-sample-pair">
        <figure v-if="s.input" class="nest-sample-figure">
          <img :src="s.input" alt="Input" loading="lazy" />
          <figcaption>Input</figcaption>
        </figure>
        <figure v-if="s.output" class="nest-sample-figure">
          <img :src="s.output" alt="Output" loading="lazy" />
          <figcaption>Output{{ s.backend ? `, ${s.backend}` : "" }}</figcaption>
        </figure>
      </div>
      <p v-if="s.caption" class="nest-sample-note">{{ s.caption }}</p>
      <dl v-if="s.psnr_db" class="nest-sample-facts">
        <template v-for="[k, v] in psnrRows(s)" :key="k"><dt>PSNR, {{ k }}</dt><dd>{{ v }}</dd></template>
        <template v-if="s.reference"><dt>Reference</dt><dd>{{ s.reference }}</dd></template>
      </dl>
    </template>

    <!-- text to image -->
    <template v-else-if="kind === 'text-to-image'">
      <div class="nest-sample-prompt">
        <span class="nest-sample-label">Prompt</span>
        <p>{{ s.prompt }}</p>
      </div>
      <div class="nest-sample-pair">
        <figure class="nest-sample-figure">
          <img :src="s.output" alt="Generated with the Linnet UNet" loading="lazy" />
          <figcaption><i class="nest-swatch is-linnet"></i>Linnet{{ s.backend ? `, ${s.backend}` : "" }}</figcaption>
        </figure>
        <figure v-if="s.reference_output" class="nest-sample-figure">
          <img :src="s.reference_output" alt="Generated with the stock pipeline" loading="lazy" />
          <figcaption><i class="nest-swatch is-reference"></i>{{ s.reference ?? "Reference" }}</figcaption>
        </figure>
      </div>
      <p v-if="s.caption" class="nest-sample-note">{{ s.caption }}</p>
      <dl class="nest-sample-facts">
        <template v-for="[k, v] in psnrRows(s)" :key="k"><dt>PSNR, {{ k }}</dt><dd>{{ v }}</dd></template>
        <template v-if="s.seconds?.linnet != null"><dt>Wall time, Linnet</dt><dd>{{ s.seconds.linnet }} s</dd></template>
        <template v-if="s.seconds?.diffusers != null"><dt>Wall time, diffusers</dt><dd>{{ s.seconds.diffusers }} s</dd></template>
      </dl>
    </template>

    <!-- transcription -->
    <template v-else-if="kind === 'transcription'">
      <p class="nest-sample-lede">One clip{{ s.audio ? ` (${s.audio})` : "" }}, transcribed greedily through Linnet and through the reference stack.</p>
      <div class="nest-sample-pair">
        <figure class="nest-sample-text">
          <figcaption><i class="nest-swatch is-linnet"></i>Linnet</figcaption>
          <p>{{ s.text }}</p>
        </figure>
        <figure v-if="s.reference_text != null" class="nest-sample-text">
          <figcaption><i class="nest-swatch is-reference"></i>{{ s.reference_stack ?? "transformers (reference)" }}</figcaption>
          <p>{{ s.reference_text }}</p>
        </figure>
      </div>
      <p v-if="sameText != null" class="nest-sample-note">{{ sameText ? "The two transcripts are identical." : "The two transcripts differ." }}</p>
    </template>

    <!-- a kind this page does not know yet: say so, and show it as recorded -->
    <template v-else>
      <p class="nest-empty-state">This sample is of a kind (<code>{{ kind || "none" }}</code>) the page has no view for yet; here it is as the harness recorded it.</p>
      <pre class="nest-snippet">{{ JSON.stringify(s, null, 2) }}</pre>
    </template>
  </div>
</template>
