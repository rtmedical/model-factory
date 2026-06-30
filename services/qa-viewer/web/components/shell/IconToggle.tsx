"use client";

// Square toggle button with the same hover / active styling used across
// the workspace top-bar, viewer stage, and fullscreen toolbar. Promoted
// out of ViewerStage so the fullscreen layer can reuse it verbatim.

import { cn } from "@/lib/utils";

export type IconToggleProps = {
  onClick: () => void;
  label: string;
  Icon: React.ComponentType<{ size?: number }>;
  active?: boolean;
  disabled?: boolean;
  /** Slightly larger pill — used in the fullscreen toolbar. */
  size?: "sm" | "md";
};

export function IconToggle({
  onClick,
  label,
  Icon,
  active = false,
  disabled = false,
  size = "sm",
}: IconToggleProps) {
  const dim = size === "md" ? "h-8 w-8" : "h-7 w-7";
  const iconSize = size === "md" ? 15 : 13;
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={label}
      aria-label={label}
      aria-pressed={active}
      className={cn(
        "inline-flex items-center justify-center rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] transition-colors",
        dim,
        disabled
          ? "cursor-not-allowed text-[var(--color-rt-muted)] opacity-50"
          : active
            ? "bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
            : "text-[var(--color-rt-muted)] hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]",
      )}
    >
      <Icon size={iconSize} />
    </button>
  );
}
