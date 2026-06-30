"use client";

import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

const RATIO = 1871 / 1064;

/**
 * Real RT Medical Systems mark. Plain <img> so it works in static export
 * without next/image's optimizer.
 *   Light theme → black-on-transparent variant (rt-medical-on-light.png)
 *   Dark  theme → white-on-transparent variant (rt-medical-on-dark.png)
 */
export function LogoMark({
  height = 40,
  className,
}: {
  height?: number;
  className?: string;
}) {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const sync = () => {
      const t = (document.documentElement.getAttribute("data-theme") as
        | "light"
        | "dark") ?? "light";
      setTheme(t);
    };
    sync();
    const obs = new MutationObserver(sync);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => obs.disconnect();
  }, []);

  const src =
    theme === "dark"
      ? "/brand/rt-medical-on-dark.png"
      : "/brand/rt-medical-on-light.png";
  const width = Math.round(height * RATIO);

  // eslint-disable-next-line @next/next/no-img-element
  return (
    <img
      src={src}
      alt="RT Medical Systems"
      width={width}
      height={height}
      draggable={false}
      className={cn("select-none", className)}
    />
  );
}

export function LogoLockup({
  height = 36,
  caption = "Model QA",
  subtitle = "Training snapshot viewer",
  className,
}: {
  height?: number;
  caption?: string;
  subtitle?: string;
  className?: string;
}) {
  // Two-tier lockup: company mark on the left, a tight stacked wordmark
  // on the right (title + subtitle) that gives the page its identity
  // without a thin vertical-rule divider — reads more professional /
  // less "dev tool" than the old chip pattern.
  return (
    <span className={cn("inline-flex items-center gap-3", className)}>
      <LogoMark height={height} />
      <span className="hidden flex-col leading-none md:inline-flex">
        <span className="rt-display text-[13px] font-semibold tracking-[0.18em] uppercase text-[var(--color-rt-ink)]">
          {caption}
        </span>
        <span className="mt-1 text-[10.5px] tracking-[0.08em] text-[var(--color-rt-muted)]">
          {subtitle}
        </span>
      </span>
    </span>
  );
}
