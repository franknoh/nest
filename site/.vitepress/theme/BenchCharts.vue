<script setup lang="ts">
// The benchmarks tab: `models/<name>/bench.json` from `bench/run.py`. First
// what the model's numbers show -- Linnet against the stack it replaces on
// each runtime (`bench/compare.json`) -- then one horizontal bar chart per
// metric, rows grouped by where they run, each runtime's stacks above
// Linnet's. Linnet rows are dark gray, the others light gray, the best bar of
// each chart carries the accent, and a table below holds every raw number.
import comparison from "@nest/compare.json";
import { computed, onBeforeUnmount, onMounted, ref, useId } from "vue";
import { type Compare, against, evaluate, grouped, label as shortLabel, percent, primary, times as speed } from "./compare";

const compare = comparison as unknown as Compare;

interface Method {
  method: string;
  kind: string;
  metrics: Record<string, number | null | undefined>;
  max_abs_diff: number | null;
  notes: string;
  error: string | null;
  key: string;
}

interface Bench {
  environment?: Record<string, string>;
  workload?: Record<string, unknown>;
  reference?: string;
  methods?: Method[];
  date?: string;
}

const props = defineProps<{ data: Bench | null; title: string }>();

// Every metric the families report, in the order the charts appear.
interface Metric {
  key: string;
  label: string;
  unit: string;
  lower: boolean; // lower is better (a time); otherwise a rate
}
const METRICS: Metric[] = [
  { key: "ttft_ms", label: "Time to first token", unit: "ms", lower: true },
  { key: "decode_tok_s", label: "Decode speed", unit: "tok/s", lower: false },
  { key: "latency_ms", label: "Latency", unit: "ms", lower: true },
  { key: "throughput_per_s", label: "Throughput", unit: "/s", lower: false },
  { key: "step_ms", label: "Denoising step", unit: "ms", lower: true },
  { key: "encode_ms", label: "Encode", unit: "ms", lower: true },
  { key: "transcribe_ms", label: "Transcribe", unit: "ms", lower: true },
  { key: "serve_tok_s", label: "Serving throughput", unit: "tok/s", lower: false },
  { key: "serve_ttft_ms", label: "Serving time to first token", unit: "ms", lower: true },
  { key: "load_s", label: "Load time", unit: "s", lower: true },
];
const MEMORY = "peak_vram_mib";
// Values a method records for the harness's own use, not for the page.
const INTERNAL = new Set(["first_token", "serve_s", "driver_vram_mib"]);

// Serving rows (`serve-*`) measure many requests at once, the others one at a
// time; a chart shows the rows that measured its metric, and a failed row
// where its kind of measurement is charted, never "not measured" filler.
function serving(key: string): boolean {
  return key.startsWith("serve");
}

const methods = computed<Method[]>(() =>
  (props.data?.methods ?? []).map((m) => ({
    method: String(m.method ?? "?"),
    // A method that crashed comes back as kind "unknown" under its key
    // (`linnet-jax`); the key still says whose it is.
    kind: m.kind === "linnet" || (m.kind !== "reference" && /^linnet/i.test(String(m.method))) ? "linnet" : "reference",
    metrics: m.metrics ?? {},
    max_abs_diff: typeof m.max_abs_diff === "number" ? m.max_abs_diff : null,
    notes: m.notes ?? "",
    error: m.error ?? null,
    key: m.key ?? "",
  })),
);
const hasData = computed(() => methods.value.length > 0);

// `reference` names the method by its key, which every row carries; a
// reference that failed falls back to the first reference row that ran.
const referenceIndex = computed(() => {
  const key = props.data?.reference ?? "";
  const rows = methods.value;
  let index = rows.findIndex((m) => m.key === key && !m.error);
  if (index < 0) index = rows.findIndex((m) => m.kind === "reference" && !m.error);
  return index;
});
const referenceName = computed(() => methods.value[referenceIndex.value]?.method ?? props.data?.reference ?? "the reference");

function value(m: Method, key: string): number | null {
  const v = m.metrics[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// ---- formatting

function num(v: number): string {
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return v.toFixed(2);
}
function raw(v: number | null, unit: string): string {
  if (v == null) return "not measured";
  if (unit === "/s") return `${num(v)}/s`;
  if (unit === "GiB") return `${v.toFixed(v >= 10 ? 1 : 2)} GiB`;
  return `${num(v)} ${unit}`;
}
function times(s: number): string {
  if (s >= 10) return `${s.toFixed(0)}x`;
  if (s < 0.1) return `${s.toFixed(2)}x`;
  return `${s.toFixed(1)}x`;
}
function distance(m: Method, index: number): string {
  if (index === referenceIndex.value) return "0 (the reference)";
  if (m.max_abs_diff == null) return "not compared";
  const d = m.max_abs_diff;
  if (d === 0) return "0";
  if (d < 1e-3) return d.toExponential(1);
  return d.toPrecision(3).replace(/\.?0+$/, "");
}
function capped(m: Method): boolean {
  return m.key === "linnet-offload";
}
// vLLM, SGLang, and TGI reserve most of the GPU up front, whichever row
// starts them (their own, Triton's backend, a Linnet export served by them).
function reserved(m: Method): boolean {
  return /vllm|sglang|tgi/.test(m.key) || /reserv\w* .*pool|pool .*reserv|gpu_memory_utilization/i.test(m.notes);
}

// ---- the charts

interface Row {
  index: number;
  method: Method;
  bar: number | null; // what the bar's length encodes
  label: string; // at the bar's end
  raw: string;
  extra: string; // the tooltip's second fact
  hatched: boolean;
  best?: boolean; // the chart's best value, drawn in the accent
  head: string | null; // the runtime this row opens, above it
}
interface Chart {
  key: string;
  title: string;
  subtitle: string;
  ratio: boolean;
  rows: Row[];
}

// Every row's place: its runtime (in `compare.json`'s order) and its place
// within it, so each chart reads runtime by runtime, the stacks first.
const order = computed(() => {
  const place = new Map<string, { group: string; rank: number }>();
  let rank = 0;
  for (const g of grouped(compare, methods.value)) for (const m of g.rows) place.set(m.key, { group: g.group.title, rank: rank++ });
  return place;
});
function arrange(items: { m: Method; index: number }[]): { m: Method; index: number; head: string | null }[] {
  const place = order.value;
  const sorted = [...items].sort((a, b) => (place.get(a.m.key)?.rank ?? 1e9) - (place.get(b.m.key)?.rank ?? 1e9));
  return sorted.map((item, i) => {
    const group = place.get(item.m.key)?.group ?? "Other";
    const before = i ? place.get(sorted[i - 1].m.key)?.group ?? "Other" : null;
    return { ...item, head: group !== before ? group : null };
  });
}
// What a row is measured against: for a Linnet row, the stack it replaces on
// its runtime; otherwise, or where that stack has no number, the reference.
function comparedWith(m: Method, index: number, metric: Metric, base: number | null): string {
  const v = value(m, metric.key);
  if (index === referenceIndex.value) return "the reference";
  const vs = against(compare, methods.value, m, metric.key);
  if (vs) {
    const them = shortLabel(compare, vs.them);
    return vs.parity ? `${percent(vs.speedup)} from ${them}` : `${speed(vs.speedup)} the speed of ${them}`;
  }
  if (v != null && v > 0 && base != null && base > 0) return `${times(metric.lower ? base / v : v / base)} the reference's speed`;
  return "";
}

const charts = computed<Chart[]>(() => {
  const out: Chart[] = [];
  const rows = methods.value;
  const ref = rows[referenceIndex.value];
  for (const metric of METRICS) {
    if (!rows.some((m) => value(m, metric.key) != null)) continue;
    const shown = arrange(
      rows
        .map((m, index) => ({ m, index }))
        .filter(({ m }) => value(m, metric.key) != null || (m.error != null && serving(m.key) === serving(metric.key) && metric.key !== "load_s")),
    );
    // Raw values, so a bar's length is the number itself and the red bar is
    // the shortest where lower is better and the longest where higher is;
    // how a row compares is in the tooltip.
    const base = ref ? value(ref, metric.key) : null;
    const direction = metric.lower ? "lower is better" : "higher is better";
    out.push({
      key: metric.key,
      title: metric.label,
      subtitle: `${metric.unit}, ${direction}; the best bar is red.`,
      ratio: false,
      rows: shown.map(({ m, index, head }) => {
        const v = value(m, metric.key);
        return {
          index,
          method: m,
          bar: v,
          label: v == null ? (m.error ? "failed" : "not measured") : raw(v, metric.unit),
          raw: raw(v, metric.unit),
          extra: comparedWith(m, index, metric, base),
          hatched: false,
          head,
        };
      }),
    });
  }
  // Two rows' memory is not a measurement to compare: vLLM reserves most of
  // the GPU for its cache pool before it runs, and the offloaded row is held
  // under a cap on purpose. They stay in the table, out of the chart.
  const unlike = (m: Method) => reserved(m) || capped(m);
  if (rows.some((m) => value(m, MEMORY) != null && !unlike(m))) {
    const left = rows.filter((m) => value(m, MEMORY) != null && unlike(m)).map((m) => shortLabel(compare, m));
    out.push({
      key: MEMORY,
      title: "Peak GPU memory",
      subtitle:
        "What the driver reports the process holding at its peak, in GiB; lower is better; the best bar is red." +
        (left.length ? ` Not drawn, since theirs is a setting rather than a need: ${left.join(", ")} (in the table).` : ""),
      ratio: false,
      rows: arrange(
        rows.map((m, index) => ({ m, index })).filter(({ m }) => value(m, MEMORY) != null && !unlike(m)),
      ).map(({ m, index, head }) => {
        const mib = value(m, MEMORY);
        const gib = mib == null ? null : mib / 1024;
        const pool = gib != null && reserved(m);
        return {
          index,
          method: m,
          bar: gib,
          label: gib == null ? (m.error ? "failed" : "not measured") : raw(gib, "GiB"),
          raw: gib == null ? "not measured" : `${raw(gib, "GiB")} (${num(mib!)} MiB)`,
          extra: pool ? "a reserved pool: this is a setting, not what the model needs" : "",
          hatched: pool,
          head,
        };
      }),
    });
  }
  // The best bar of each chart in the accent: a speed-up is always higher
  // is better, a raw value goes the metric's way, memory lower. A reserved
  // pool is a setting, not a result, so it never wins memory.
  for (const chart of out) {
    const metric = METRICS.find((m) => m.key === chart.key);
    const lower = chart.key === MEMORY || (metric?.lower ?? false);
    // The offloaded row runs under a memory cap by design: it shows what a
    // small GPU can do, and wins nothing against unconstrained rows.
    const candidates = chart.rows.filter((r) => r.bar != null && !r.hatched && !r.method.error && !capped(r.method));
    if (!candidates.length) continue;
    const bars = candidates.map((r) => r.bar as number);
    const target = lower ? Math.min(...bars) : Math.max(...bars);
    for (const r of candidates) r.best = r.bar === target;
  }
  return out;
});
const anyReserved = computed(() => charts.value.some((c) => c.rows.some((r) => r.hatched)));

// ---- geometry: drawn at the container's own width, so text stays its size

const box = ref<HTMLElement | null>(null);
const width = ref(640);
let observer: ResizeObserver | null = null;
onMounted(() => {
  if (!box.value) return;
  width.value = Math.max(240, Math.round(box.value.clientWidth));
  observer = new ResizeObserver((entries) => {
    const w = Math.round(entries[0].contentRect.width);
    if (w > 0) width.value = Math.max(240, w);
  });
  observer.observe(box.value);
});
onBeforeUnmount(() => observer?.disconnect());

const BAR = 18;
const GAP = 8;
const LINE = 15;
const AXIS = 22;
const END = 64; // room for the label at a bar's end
const CHAR = 6.9; // average glyph width at 12.5px Inter

const narrow = computed(() => width.value < 520);
const labelWidth = computed(() => (narrow.value ? width.value : Math.min(240, Math.round(width.value * 0.36))));
const plotLeft = computed(() => (narrow.value ? 0 : labelWidth.value + 12));
const plotRight = computed(() => width.value - END);

// A method name in at most two lines of the label column; the tooltip and
// the table carry it whole.
function wrap(text: string): string[] {
  const max = Math.max(8, Math.floor(labelWidth.value / CHAR));
  const words = text.split(/\s+/);
  const lines: string[] = [];
  let line = "";
  for (const word of words) {
    if (!line) line = word;
    else if ((line + " " + word).length <= max) line += " " + word;
    else {
      lines.push(line);
      line = word;
    }
  }
  if (line) lines.push(line);
  const shown = lines.slice(0, 2).map((l) => (l.length > max ? l.slice(0, max - 1) + "…" : l));
  if (lines.length > 2) shown[1] = shown[1].slice(0, Math.max(1, max - 1)).replace(/\s*\S?$/, "") + "…";
  return shown;
}

function niceStep(span: number): number {
  const rough = span / 4;
  const power = 10 ** Math.floor(Math.log10(rough));
  for (const f of [1, 2, 2.5, 5, 10]) if (f * power >= rough) return f * power;
  return 10 * power;
}

interface Layout {
  height: number;
  top: (i: number) => number; // row's top edge
  bar: (i: number) => number; // bar's top edge
  pitch: (i: number) => number;
  lines: string[][];
  x: (v: number) => number;
  ticks: number[];
  plotBottom: number;
  heads: { title: string; y: number }[]; // each runtime's title, above its rows
}

const HEAD = 26;

function layout(chart: Chart): Layout {
  const lines = chart.rows.map((r) => wrap(shortLabel(compare, r.method)));
  const pitches = lines.map((l) => (narrow.value ? l.length * LINE + 4 + BAR + GAP + 2 : Math.max(BAR, l.length * LINE) + GAP));
  const tops: number[] = [];
  const heads: { title: string; y: number }[] = [];
  let y = 4;
  pitches.forEach((p, i) => {
    const head = chart.rows[i].head;
    if (head) {
      heads.push({ title: head, y });
      y += HEAD;
    }
    tops.push(y);
    y += p;
  });
  const plotBottom = y;
  const max = Math.max(chart.ratio ? 1 : 0, ...chart.rows.map((r) => r.bar ?? 0)) || 1;
  const step = niceStep(max);
  const domain = Math.max(max * 1.04, step);
  const ticks: number[] = [];
  for (let t = 0; t <= domain + 1e-9; t += step) ticks.push(Number(t.toPrecision(6)));
  const left = plotLeft.value;
  const span = Math.max(40, plotRight.value - left);
  return {
    height: plotBottom + AXIS,
    top: (i) => tops[i],
    pitch: (i) => pitches[i],
    bar: (i) => (narrow.value ? tops[i] + lines[i].length * LINE + 4 : tops[i] + (pitches[i] - GAP - BAR) / 2),
    lines,
    x: (v) => left + (v / domain) * span,
    ticks,
    plotBottom,
    heads,
  };
}
const layouts = computed(() => charts.value.map(layout));

// A bar with a square baseline and a 4px rounded data end.
function barPath(x0: number, x1: number, y: number): string {
  const w = Math.max(0, x1 - x0);
  const r = Math.min(4, w / 2, BAR / 2);
  return `M${x0},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${BAR - 2 * r}a${r},${r} 0 0 1 ${-r},${r}h${-(w - r)}z`;
}
function tickLabel(chart: Chart, t: number): string {
  if (chart.ratio) return `${Number(t.toPrecision(3))}x`;
  return `${Number(t.toPrecision(4))}`;
}
// The 1.0x rule gets its own axis label when no tick lands on it and there is room.
function ruleLabel(l: Layout): boolean {
  if (l.ticks.includes(1)) return false;
  return l.ticks.every((t) => Math.abs(l.x(t) - l.x(1)) > 26);
}
function unitOf(chart: Chart): string {
  if (chart.key === MEMORY) return "GiB";
  return METRICS.find((m) => m.key === chart.key)?.unit ?? "";
}

// ---- hover: one row at a time, the whole row is the target

const active = ref<{ chart: string; row: number } | null>(null);
function show(chart: string, row: number) {
  active.value = { chart, row };
}
function hide(chart: string, row: number) {
  if (active.value?.chart === chart && active.value.row === row) active.value = null;
}
function tipStyle(ci: number, ri: number) {
  const l = layouts.value[ci];
  const top = l.top(ri) + l.pitch(ri) - 2;
  const left = narrow.value ? 0 : Math.min(plotLeft.value, Math.max(0, width.value - 320));
  return { top: `${top}px`, left: `${left}px` };
}
function rowLabel(chart: Chart, row: Row): string {
  const parts = [row.method.method, row.raw];
  if (chart.ratio && row.extra) parts.push(row.extra);
  if (!chart.ratio && row.extra) parts.push(row.extra);
  return parts.join(", ");
}

const uid = useId();

// ---- context: what, on what, and when

const context = computed(() => {
  const env = props.data?.environment ?? {};
  const work = (props.data?.workload ?? {}) as Record<string, unknown>;
  const parts: string[] = [];
  if (env.gpu && env.gpu !== "none") {
    const count = Number(env.gpus ?? 1);
    parts.push(count > 1 ? `${count} x ${env.gpu}` : env.gpu);
  } else if (env.gpu === "none") {
    parts.push("CPU only (no GPU found)");
  }
  if (work.device && !(env.gpu && env.gpu !== "none" && work.device === "cuda")) parts.push(`device ${work.device}`);
  const text = methods.value.some((m) => value(m, "ttft_ms") != null || value(m, "decode_tok_s") != null);
  if (text && work.prompt_tokens != null) parts.push(`${work.prompt_tokens} prompt tokens, ${work.new_tokens} new`);
  if (text && work.batch != null) parts.push(`batch ${work.batch}`);
  for (const key of ["dtype", "precision"]) if (work[key]) parts.push(String(work[key]));
  if (work.iters != null) parts.push(`median of ${work.iters} runs after ${work.warmup ?? 0} warm-ups`);
  if (props.data?.date) parts.push(`measured ${props.data.date}`);
  return parts.join(" · ");
});
const software = computed(() => {
  const env = props.data?.environment ?? {};
  const names: [string, string][] = [
    ["linnet", "Linnet"],
    ["torch", "PyTorch"],
    ["jax", "JAX"],
    ["transformers", "transformers"],
    ["vllm", "vLLM"],
    ["diffusers", "diffusers"],
    ["onnxruntime", "ONNX Runtime"],
    ["python", "Python"],
  ];
  const parts = names.filter(([key]) => env[key]).map(([key, name]) => `${name} ${env[key]}`);
  if (env.driver) parts.push(`driver ${env.driver}`);
  return parts.join(", ");
});

// ---- what the numbers show: Linnet against the stack it replaces, on each
// runtime and in each engine the model takes part in

const verdicts = computed(() =>
  compare.matchups.flatMap((matchup) => {
    const r = evaluate(matchup, methods.value);
    if (!r) return [];
    const s = r.lead.speedup;
    const metric = METRICS.find((m) => m.key === r.metric);
    const unit = metric?.unit ?? "";
    const parity = !!matchup.parity;
    return [
      {
        id: matchup.id,
        title: matchup.short,
        number: parity ? percent(s) : speed(s),
        word: parity ? (Math.abs(s - 1) < 0.03 ? "the same speed" : s > 1 ? "faster" : "slower") : s >= 1 ? "faster" : "slower",
        // An accent where Linnet is ahead, gray where it is behind, and
        // Linnet's own gray where the two are even.
        outcome: parity || Math.abs(s - 1) < 0.03 ? "even" : s > 1 ? "ahead" : "behind",
        measure: (metric?.label ?? r.metric).toLowerCase(),
        lines: r.pairs.map((pair) => ({
          us: shortLabel(compare, pair.us),
          usValue: raw(pair.usValue, unit),
          them: shortLabel(compare, pair.them),
          themValue: raw(pair.themValue, unit),
        })),
      },
    ];
  }),
);

// ---- the table: every method, every metric, raw

const tableMetrics = computed(() => {
  const keys = new Set<string>();
  for (const m of methods.value) for (const [k, v] of Object.entries(m.metrics)) if (v != null && !INTERNAL.has(k)) keys.add(k);
  const known = [...METRICS.map((m) => m.key), MEMORY].filter((k) => keys.has(k));
  const rest = [...keys].filter((k) => !known.includes(k)).sort();
  return [...known, ...rest].map((key) => {
    const metric = METRICS.find((m) => m.key === key);
    if (key === MEMORY) return { key, label: "Peak GPU memory", unit: "GiB" };
    return { key, label: metric?.label ?? key, unit: metric?.unit ?? "" };
  });
});
// The table's rows by runtime, and each Linnet row against the stack it
// replaces, on the model's own measure (serving rows on theirs).
const tableGroups = computed(() => grouped(compare, methods.value.map((m, index) => ({ ...m, index }))));
const mainMetric = computed(() => primary(methods.value));
function versus(m: Method): string {
  const metric = serving(m.key) ? "serve_tok_s" : mainMetric.value;
  const vs = metric ? against(compare, methods.value, m, metric) : null;
  if (!vs) return "";
  const them = shortLabel(compare, vs.them);
  return vs.parity ? `${percent(vs.speedup)} from ${them}` : `${speed(vs.speedup)} ${them}`;
}
function cell(m: Method, key: string, unit: string): string {
  const v = value(m, key);
  if (v == null) return "–";
  if (key === MEMORY) return raw(v / 1024, "GiB");
  return unit ? raw(v, unit) : num(v);
}
</script>

<template>
  <div class="nest-bench" ref="box">
    <p v-if="!hasData" class="nest-empty-state">
      {{ title }} has not been measured yet. Its numbers will appear here once the benchmark harness has run it on a GPU against the stacks people already use.
    </p>
    <template v-else>
      <p class="nest-bench-context">{{ context }}</p>
      <p v-if="software" class="nest-bench-software">{{ software }}</p>

      <h3 v-if="verdicts.length" class="nest-verdicts-title">Linnet against the stack it replaces</h3>
      <p v-if="verdicts.length" class="nest-bench-sub">
        Each side in its fastest configuration, on the same GPU and checkpoint. A speed-up is how many times the other's speed; an export is measured
        against the original checkpoint in the same engine.
      </p>
      <section v-if="verdicts.length" class="nest-verdicts" aria-label="Linnet against the stack it replaces">
        <div v-for="v in verdicts" :key="v.id" class="nest-verdict" :class="`is-${v.outcome}`">
          <span class="nest-verdict-title">{{ v.title }}</span>
          <span class="nest-verdict-number"><strong>{{ v.number }}</strong> {{ v.word }}</span>
          <span class="nest-verdict-measure">{{ v.measure }}</span>
          <ul>
            <li v-for="(l, i) in v.lines" :key="i">
              <span class="is-linnet"><span>{{ l.us }}</span><span>{{ l.usValue }}</span></span>
              <span class="is-them"><span>{{ l.them }}</span><span>{{ l.themValue }}</span></span>
            </li>
          </ul>
        </div>
      </section>

      <div class="nest-bench-legend" aria-hidden="true">
        <span><i class="nest-swatch is-best"></i>Best in the chart</span>
        <span><i class="nest-swatch is-linnet"></i>Linnet</span>
        <span><i class="nest-swatch is-reference"></i>Reference</span>
        <span v-if="anyReserved"><i class="nest-swatch is-reserved"></i>Reserved pool (a setting, not a need)</span>
      </div>
      <p class="nest-bench-note">
        Each row also carries its <strong>distance from reference</strong>: the largest absolute difference between its output and {{ referenceName }}'s on the same input.
        bf16 outputs differ by rounding (about 0.1 on logits near 16), so a small number is expected; it is there so that a fast wrong answer cannot look like a win.
      </p>

      <section v-for="(chart, ci) in charts" :key="chart.key" class="nest-bench-chart">
        <h3>{{ chart.title }}</h3>
        <p class="nest-bench-sub">{{ chart.subtitle }}</p>
        <div class="nest-bench-plot">
          <svg
            :viewBox="`0 0 ${width} ${layouts[ci].height}`"
            :width="width"
            :height="layouts[ci].height"
            role="group"
            :aria-label="`${chart.title}: ${chart.subtitle}`"
          >
            <defs v-if="chart.key === 'peak_vram_mib'">
              <pattern v-for="kind in ['linnet', 'reference']" :key="kind" :id="`${uid}-hatch-${kind}`" patternUnits="userSpaceOnUse" width="5" height="5" patternTransform="rotate(45)">
                <line x1="0" y1="0" x2="0" y2="5" :class="`nest-bench-hatch is-${kind}`" />
              </pattern>
            </defs>
            <g class="nest-bench-grid">
              <line v-for="t in layouts[ci].ticks" :key="t" :x1="layouts[ci].x(t)" :x2="layouts[ci].x(t)" y1="0" :y2="layouts[ci].plotBottom" />
              <line class="nest-bench-baseline" :x1="layouts[ci].x(0)" :x2="layouts[ci].x(0)" y1="0" :y2="layouts[ci].plotBottom" />
            </g>
            <g v-for="h in layouts[ci].heads" :key="h.title" class="nest-bench-head">
              <line x1="0" :x2="width" :y1="h.y + 4" :y2="h.y + 4" />
              <text x="0" :y="h.y + 18">{{ h.title }}</text>
            </g>
            <g class="nest-bench-axis">
              <text v-for="t in layouts[ci].ticks" :key="t" :x="layouts[ci].x(t)" :y="layouts[ci].plotBottom + 15" text-anchor="middle">{{ tickLabel(chart, t) }}</text>
              <text v-if="!chart.ratio" :x="plotRight + 6" :y="layouts[ci].plotBottom + 15">{{ unitOf(chart) }}</text>
              <text v-if="chart.ratio && ruleLabel(layouts[ci])" class="nest-bench-rule-label" :x="layouts[ci].x(1)" :y="layouts[ci].plotBottom + 15" text-anchor="middle">1x</text>
            </g>
            <g
              v-for="(row, ri) in chart.rows"
              :key="ri"
              class="nest-bench-row"
              :class="{ active: active?.chart === chart.key && active.row === ri }"
              tabindex="0"
              :aria-label="rowLabel(chart, row)"
              @mouseenter="show(chart.key, ri)"
              @mouseleave="hide(chart.key, ri)"
              @focus="show(chart.key, ri)"
              @blur="hide(chart.key, ri)"
              @click="show(chart.key, ri)"
            >
              <rect class="nest-bench-hit" x="0" :y="layouts[ci].top(ri) - GAP / 2" :width="width" :height="layouts[ci].pitch(ri)" />
              <text
                class="nest-bench-label"
                :x="narrow ? 0 : labelWidth"
                :text-anchor="narrow ? 'start' : 'end'"
                :y="narrow ? layouts[ci].top(ri) + 11 : layouts[ci].bar(ri) + BAR / 2 - ((layouts[ci].lines[ri].length - 1) * LINE) / 2 + 4"
              >
                <tspan v-for="(line, li) in layouts[ci].lines[ri]" :key="li" :x="narrow ? 0 : labelWidth" :dy="li ? LINE : 0">{{ line }}</tspan>
              </text>
              <path
                v-if="row.bar != null"
                class="nest-bench-bar"
                :class="[`is-${row.method.kind}`, { 'is-hatched': row.hatched, 'is-best': row.best }]"
                :style="row.hatched ? { fill: `url(#${uid}-hatch-${row.method.kind})` } : undefined"
                :d="barPath(layouts[ci].x(0), layouts[ci].x(row.bar), layouts[ci].bar(ri))"
              />
              <text
                class="nest-bench-value"
                :class="{ 'is-missing': row.bar == null }"
                :x="(row.bar != null ? layouts[ci].x(row.bar) : layouts[ci].x(0)) + 6"
                :y="layouts[ci].bar(ri) + BAR / 2 + 4"
              >{{ row.label }}</text>
            </g>
            <g v-if="chart.ratio" class="nest-bench-rule">
              <line :x1="layouts[ci].x(1)" :x2="layouts[ci].x(1)" y1="0" :y2="layouts[ci].plotBottom" />
            </g>
          </svg>
          <div
            v-for="(row, ri) in chart.rows"
            v-show="active?.chart === chart.key && active.row === ri"
            :key="ri"
            class="nest-bench-tip"
            :style="tipStyle(ci, ri)"
            role="tooltip"
          >
            <strong>{{ row.method.method }}</strong>
            <span v-if="row.method.error" class="nest-bench-tip-error">Failed: {{ row.method.error }}</span>
            <template v-else>
              <span>{{ row.raw }}</span>
              <span v-if="row.extra" :class="{ 'nest-bench-tip-flag': row.hatched }">{{ row.extra }}</span>
              <span>Distance from reference: {{ distance(row.method, row.index) }}</span>
              <span v-if="row.method.notes" class="nest-bench-tip-notes">{{ row.method.notes }}</span>
            </template>
          </div>
        </div>
      </section>

      <h3 class="nest-bench-table-title">Every number</h3>
      <div class="nest-bench-table-wrap">
        <table class="nest-bench-table">
          <thead>
            <tr>
              <th>Method</th>
              <th v-for="m in tableMetrics" :key="m.key">{{ m.label }}</th>
              <th>Against the stack it replaces</th>
              <th>Distance from reference</th>
              <th>Notes</th>
            </tr>
          </thead>
          <tbody v-for="g in tableGroups" :key="g.group.id">
            <tr class="nest-bench-group">
              <th :colspan="tableMetrics.length + 4" scope="rowgroup">{{ g.group.title }}</th>
            </tr>
            <tr v-for="m in g.rows" :key="m.index" :class="{ 'is-error': m.error }">
              <th scope="row" :title="m.method"><i class="nest-swatch" :class="`is-${m.kind}`"></i>{{ shortLabel(compare, m) }}</th>
              <td v-for="t in tableMetrics" :key="t.key" class="num">{{ cell(m, t.key, t.unit) }}</td>
              <td class="num">{{ m.error ? "–" : versus(m) }}</td>
              <td class="num">{{ m.error ? "–" : distance(m, m.index) }}</td>
              <td class="notes">
                <span v-if="m.error" class="nest-bench-error">Failed: {{ m.error }}</span>
                <span v-else>{{ m.notes }}</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>
