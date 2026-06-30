"use client";

// One-time initialization of Cornerstone3D + tools + the NIfTI volume loader.
// Called by the viewer component on mount; safe to call repeatedly (guarded
// by an `_initialized` flag).

import * as cornerstone from "@cornerstonejs/core";
import * as cornerstoneTools from "@cornerstonejs/tools";
import { cornerstoneNiftiImageVolumeLoader } from "@cornerstonejs/nifti-volume-loader";

let _initialized: Promise<void> | null = null;

export function ensureCornerstone(): Promise<void> {
  if (_initialized) return _initialized;
  _initialized = (async () => {
    await cornerstone.init();
    await cornerstoneTools.init();

    cornerstone.volumeLoader.registerVolumeLoader(
      "nifti",
      // The library exports the loader factory under this name.
      // The signature is (volumeId, options) -> { promise }.
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cornerstoneNiftiImageVolumeLoader as any,
    );

    // Standard navigation tools.
    //
    // The MouseBindings enum and the bindings-shape of `setToolActive` have
    // shifted across cornerstone-tools 1.x minor releases. Cast the namespace
    // to `any` for the tool wiring — the published .d.ts lags the runtime,
    // and we accept whatever the installed version actually does.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const ct: any = cornerstoneTools;
    const {
      StackScrollTool,
      PanTool,
      ZoomTool,
      WindowLevelTool,
      SegmentationDisplayTool,
      ToolGroupManager,
      // Labelmap-editing tools (1.84). These exist on the runtime even
      // when the d.ts doesn't list them; the `any` cast above lets us
      // pull them out without a build-time error.
      BrushTool,
      RectangleScissorsTool,
      CircleScissorsTool,
      SphereScissorsTool,
      PaintFillTool,
      SegmentSelectTool,
      // Links axial/coronal/sagittal viewports in MPR fullscreen mode.
      // Only meaningful when 2+ viewports share the tool group, so it's
      // passive in the default single-viewport stage.
      CrosshairsTool,
      // 3D viewer (Volume3DStage). TrackballRotateTool drives the orbit
      // camera on mouse drag; VolumeRotateMouseWheelTool gives wheel-roll.
      // Both are version-tolerant via the tryAdd guard.
      TrackballRotateTool,
      VolumeRotateMouseWheelTool,
    } = ct;
    ct.addTool(StackScrollTool);
    ct.addTool(PanTool);
    ct.addTool(ZoomTool);
    ct.addTool(WindowLevelTool);
    // Required for any segmentation rep (labelmap/contour/surface) to
    // render. Without this tool on the group, addSegmentationRepresentations
    // throws `'SegmentationDisplay' is not registered with this toolGroup`
    // and the overlay silently never appears.
    if (SegmentationDisplayTool) {
      ct.addTool(SegmentationDisplayTool);
    }
    // Edit tools — registered up front so flipping into GT-correction
    // mode is a pure activation/binding switch, not a tool-registration.
    // Each one is wrapped in a try/catch because the toolset registered
    // on `@cornerstonejs/tools@1.84` can drift between patch releases.
    const editToolsRegistered: string[] = [];
    function tryAdd(name: string, ctor: unknown) {
      if (!ctor) return;
      try {
        ct.addTool(ctor);
        editToolsRegistered.push(name);
      } catch (err) {
        console.warn(`[cornerstoneInit] could not register ${name}:`, err);
      }
    }
    tryAdd("BrushTool", BrushTool);
    tryAdd("RectangleScissorsTool", RectangleScissorsTool);
    tryAdd("CircleScissorsTool", CircleScissorsTool);
    tryAdd("SphereScissorsTool", SphereScissorsTool);
    tryAdd("PaintFillTool", PaintFillTool);
    tryAdd("SegmentSelectTool", SegmentSelectTool);
    tryAdd("CrosshairsTool", CrosshairsTool);
    tryAdd("TrackballRotateTool", TrackballRotateTool);
    tryAdd("VolumeRotateMouseWheelTool", VolumeRotateMouseWheelTool);

    let group = ToolGroupManager.getToolGroup("qa-viewer-tools");
    if (!group) {
      group = ToolGroupManager.createToolGroup("qa-viewer-tools");
      group.addTool(StackScrollTool.toolName);
      group.addTool(PanTool.toolName);
      group.addTool(ZoomTool.toolName);
      group.addTool(WindowLevelTool.toolName);
      if (SegmentationDisplayTool) {
        group.addTool(SegmentationDisplayTool.toolName);
        // Display tools are "enabled" (rendered) without mouse bindings.
        group.setToolEnabled(SegmentationDisplayTool.toolName);
      }
      // Register edit tools on the group (no bindings — they're passive
      // until GT edit mode binds them to Primary).
      function tryGroupAdd(toolName: string | undefined) {
        if (!toolName) return;
        try {
          group.addTool(toolName);
        } catch (err) {
          console.warn(`[cornerstoneInit] could not add ${toolName} to group:`, err);
        }
      }
      tryGroupAdd(BrushTool?.toolName);
      tryGroupAdd(RectangleScissorsTool?.toolName);
      tryGroupAdd(CircleScissorsTool?.toolName);
      tryGroupAdd(SphereScissorsTool?.toolName);
      tryGroupAdd(PaintFillTool?.toolName);
      tryGroupAdd(SegmentSelectTool?.toolName);
      tryGroupAdd(CrosshairsTool?.toolName);

      const MB = ct.Enums.MouseBindings;
      group.setToolActive(WindowLevelTool.toolName, {
        bindings: [{ mouseButton: MB.Primary }],
      });
      group.setToolActive(PanTool.toolName, {
        bindings: [{ mouseButton: MB.Auxiliary }],
      });
      group.setToolActive(ZoomTool.toolName, {
        bindings: [{ mouseButton: MB.Secondary }],
      });
      // Wheel scroll for stack: prefer the explicit Wheel binding when the
      // installed enum exports it; fall back to Auxiliary on older versions.
      const wheelBinding = MB.Wheel ?? MB.Auxiliary;
      group.setToolActive(StackScrollTool.toolName, {
        bindings: [{ mouseButton: wheelBinding }],
      });
    }

    // Expose tool names + the (any-cast) ct namespace for the GT edit
    // wiring. Each entry maps the QA-store tool key to the cornerstone-tools
    // tool name string. Missing entries (older runtime) leave the
    // corresponding button disabled in the toolbar.
    // Every brush variant routes through the same BrushTool — we swap
    // the activeStrategy at bind time (see QA_BRUSH_STRATEGIES below)
    // so cornerstone paints / erases / threshold-fills with the same
    // instance, no per-variant tool registration needed.
    QA_EDIT_TOOL_NAMES = {
      brush2D: BrushTool?.toolName,
      brush3D: BrushTool?.toolName,
      eraser2D: BrushTool?.toolName,
      eraser3D: BrushTool?.toolName,
      thresholdBrush: BrushTool?.toolName,
      rectScissors: RectangleScissorsTool?.toolName,
      circleScissors: CircleScissorsTool?.toolName,
      paintFill: PaintFillTool?.toolName,
      sphereScissors: SphereScissorsTool?.toolName,
      segSelect: SegmentSelectTool?.toolName,
    };
    QA_CROSSHAIRS_TOOL_NAME = CrosshairsTool?.toolName ?? null;
    QA_BRUSH_TOOL_NAME = BrushTool?.toolName ?? null;
    QA_TRACKBALL_ROTATE_TOOL_NAME = TrackballRotateTool?.toolName ?? null;
    QA_VOLUME_ROTATE_WHEEL_TOOL_NAME = VolumeRotateMouseWheelTool?.toolName ?? null;
  })();
  return _initialized;
}

// Tool-name lookup populated by ensureCornerstone(). Keys match the
// `GtEditTool` union in lib/store.ts so the toolbar can request a tool
// by its semantic name without touching the cornerstone-tools runtime.
export let QA_EDIT_TOOL_NAMES: Partial<Record<string, string | undefined>> = {};

// `null` until ensureCornerstone() runs; also null if the installed
// cornerstone-tools build doesn't ship the tool.
export let QA_CROSSHAIRS_TOOL_NAME: string | null = null;

// The single BrushTool instance name — every brush-style edit variant
// (2D paint, 3D paint, 2D eraser, 3D eraser, threshold) routes through
// it via ToolGroup.setActiveStrategy(QA_BRUSH_TOOL_NAME, strategy).
export let QA_BRUSH_TOOL_NAME: string | null = null;

// Maps semantic edit-tool keys → BrushTool activeStrategy names. Pulled
// straight from cornerstone-tools 1.86's BrushTool source so spelling
// drift between minor releases is caught at type-check rather than at
// click time. Non-brush tools (scissors, paintFill, segSelect) are
// `null` — they map 1:1 to their own toolName.
export const QA_BRUSH_STRATEGIES: Partial<Record<string, string>> = {
  brush2D: "FILL_INSIDE_CIRCLE",
  brush3D: "FILL_INSIDE_SPHERE",
  eraser2D: "ERASE_INSIDE_CIRCLE",
  eraser3D: "ERASE_INSIDE_SPHERE",
  thresholdBrush: "THRESHOLD_INSIDE_CIRCLE",
};

export const QA_TOOL_GROUP_ID = "qa-viewer-tools";
export const QA_RENDERING_ENGINE_ID = "qa-rendering-engine";
export const QA_VIEWPORT_ID = "qa-viewport";

// MPR (fullscreen) — three additional viewports on the **same**
// rendering engine as the main viewer. Cornerstone3D creates one WebGL
// context per RenderingEngine; sharing a cached volume across engines
// fails with `bindTexture: object does not belong to this context`
// because the GPU textures are tied to the originating GL context.
// Hosting all 4 viewports on QA_RENDERING_ENGINE_ID side-steps that —
// the main viewport's host element stays mounted behind the overlay
// and just renders to an invisible canvas while MPR is up.
export const QA_MPR_VIEWPORT_IDS = {
  axial: "qa-mpr-axial",
  coronal: "qa-mpr-coronal",
  sagittal: "qa-mpr-sagittal",
} as const;

// Volume3D (fullscreen surface mesh) — single VOLUME_3D viewport, also on
// QA_RENDERING_ENGINE_ID so the prediction labelmap volume's GL textures
// are reusable for the marching-cubes pass. Uses its own tool group
// (QA_VOLUME_3D_TOOL_GROUP_ID) so trackball rotation doesn't leak into
// the 2D / MPR panes.
export const QA_VOLUME_3D_VIEWPORT_ID = "qa-volume-3d";
export const QA_VOLUME_3D_TOOL_GROUP_ID = "qa-volume-3d-tools";

// Both are `null` until ensureCornerstone has run; also null when the
// installed @cornerstonejs/tools build doesn't ship the tool.
export let QA_TRACKBALL_ROTATE_TOOL_NAME: string | null = null;
export let QA_VOLUME_ROTATE_WHEEL_TOOL_NAME: string | null = null;

// Tool names that drive the "idle" (non-edit) bindings. Keeping these
// in one place so GT-edit mode can deactivate them when it takes over
// the Primary mouse button and restore them on exit.
export const QA_IDLE_PRIMARY_TOOL = "WindowLevel";
export const QA_IDLE_AUX_TOOL = "Pan";
export const QA_IDLE_SECONDARY_TOOL = "Zoom";
