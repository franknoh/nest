import type { Theme } from "vitepress";
import Layout from "./Layout.vue";
import "./custom.css";

// A custom theme, not the documentation one: a model hub has a catalogue
// and model pages, no sidebars or prev/next links.
export default {
  Layout,
} satisfies Theme;
