// Shared dice/HD95 formatting + tier color language.
//
// Extracted from MetricsBlock so the cross-validation panel and report reuse
// the exact same thresholds and colours (no drift between the single-mode
// per-label table and the per-fold comparison).
//
// Threshold tiers (also used to colour the hero number ring):
//   >= 0.80         emerald  (--color-rt-pip-ok)
//   [0.60, 0.80)    accent   (--color-rt-accent)
//   [0.40, 0.60)    amber    (--card-amber-bg)
//   <  0.40         rose     (--color-rt-pip-error)  + leading dot

export const FAIL_THRESHOLD = 0.4;

export type Tier = "ok" | "good" | "warn" | "fail";

export function tierFor(dice: number | null): Tier {
  if (dice === null || Number.isNaN(dice)) return "fail";
  if (dice >= 0.8) return "ok";
  if (dice >= 0.6) return "good";
  if (dice >= 0.4) return "warn";
  return "fail";
}

export const TIER_VAR: Record<Tier, string> = {
  ok: "var(--color-rt-pip-ok)",
  good: "var(--color-rt-accent)",
  warn: "var(--card-amber-bg)",
  fail: "var(--color-rt-pip-error)",
};

export function formatDice(d: number | null): string {
  if (d === null || Number.isNaN(d)) return "—";
  return d.toFixed(2);
}

export function formatHd95(mm: number | null): string {
  if (mm === null || Number.isNaN(mm)) return "—";
  if (mm >= 100) return `${Math.round(mm)} mm`;
  return `${mm.toFixed(1)} mm`;
}
