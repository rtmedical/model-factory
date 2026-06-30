import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDice(d: number | null | undefined): string {
  if (d === null || d === undefined || Number.isNaN(d)) return "—";
  return d.toFixed(3);
}

export function formatElapsed(s: number): string {
  if (s < 1) return `${Math.round(s * 1000)} ms`;
  if (s < 60) return `${s.toFixed(1)} s`;
  return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
}

export function regionLabel(region: string | null | undefined): string {
  switch (region) {
    case "brain_mr":
      return "Brain MR";
    case "hn_ct":
      return "Head & Neck CT";
    case "pelvis_ct":
      return "Pelvic CT";
    case "abdomen_ct":
      return "Abdomen CT";
    case "thorax_ct":
      return "Thorax CT";
    case "whole_body_ct":
      return "Whole Body CT";
    default:
      return region ?? "—";
  }
}
