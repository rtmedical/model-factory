// Single source of truth for the contouring tool catalog. Both the
// floating GtEditToolbar (compact, lives over the main viewer) and the
// FullscreenToolbar (top strip in MPR fullscreen) render from this list
// so a click on the same tool icon in either place picks the same tool.
//
// `dim` is a small badge rendered in the corner of the button:
//   2D = single-slice operation
//   3D = volumetric (sphere) operation
//
// `group` lets us draw hairline dividers between functional clusters
// without hard-coding the layout: paint → erase → smart → fill → select.

import {
  Brush,
  Circle,
  Eraser,
  Globe,
  MousePointer2,
  PaintBucket,
  PaintbrushVertical,
  Sparkles,
  Square,
} from "lucide-react";

import type { GtEditTool } from "@/lib/store";

export type ToolEntry = {
  key: GtEditTool;
  label: string;
  hint?: string;
  Icon: React.ComponentType<{ size?: number }>;
  dim?: "2D" | "3D";
  group: "paint" | "erase" | "smart" | "fill" | "select";
};

export const TOOL_CATALOG: ToolEntry[] = [
  // Paint — most-used row, leftmost
  {
    key: "brush2D",
    label: "Brush · 2D",
    hint: "Paint the active label on the current slice only",
    Icon: Brush,
    dim: "2D",
    group: "paint",
  },
  {
    key: "brush3D",
    label: "Brush · 3D",
    hint: "Paint a sphere of voxels across multiple slices",
    Icon: PaintbrushVertical,
    dim: "3D",
    group: "paint",
  },
  // Erase — proper eraser (BrushTool with ERASE strategies), not scissors
  {
    key: "eraser2D",
    label: "Eraser · 2D",
    hint: "Erase the active label on the current slice only",
    Icon: Eraser,
    dim: "2D",
    group: "erase",
  },
  {
    key: "eraser3D",
    label: "Eraser · 3D",
    hint: "Erase a sphere of voxels across multiple slices",
    Icon: Eraser,
    dim: "3D",
    group: "erase",
  },
  // Smart — HU-threshold brush is the entry point for image-aware paint
  {
    key: "thresholdBrush",
    label: "Smart brush",
    hint: "Paint only voxels whose intensity falls inside the current HU window",
    Icon: Sparkles,
    group: "smart",
  },
  // Fill — bulk operations: 3D flood + scissor selections
  {
    key: "paintFill",
    label: "Flood fill · 3D",
    hint: "Flood the connected component starting at the clicked voxel",
    Icon: PaintBucket,
    dim: "3D",
    group: "fill",
  },
  {
    key: "rectScissors",
    label: "Rectangle fill",
    hint: "Drag a rectangle; everything inside fills with the active label",
    Icon: Square,
    dim: "2D",
    group: "fill",
  },
  {
    key: "circleScissors",
    label: "Circle fill",
    hint: "Drag a circle; everything inside fills with the active label",
    Icon: Circle,
    dim: "2D",
    group: "fill",
  },
  {
    key: "sphereScissors",
    label: "Sphere fill",
    hint: "Drag a 3D sphere; every voxel inside fills with the active label",
    Icon: Globe,
    dim: "3D",
    group: "fill",
  },
  // Select — eyedropper; switches the active segment to the one under
  // the cursor without painting.
  {
    key: "segSelect",
    label: "Pick",
    hint: "Hover over a labeled voxel to switch the active segment",
    Icon: MousePointer2,
    group: "select",
  },
];

export const TOOL_BY_KEY: Record<GtEditTool, ToolEntry> = TOOL_CATALOG.reduce(
  (acc, t) => {
    acc[t.key] = t;
    return acc;
  },
  {} as Record<GtEditTool, ToolEntry>,
);

// Group order used by the toolbars when laying out buttons.
export const TOOL_GROUP_ORDER: ToolEntry["group"][] = [
  "paint",
  "erase",
  "smart",
  "fill",
  "select",
];
