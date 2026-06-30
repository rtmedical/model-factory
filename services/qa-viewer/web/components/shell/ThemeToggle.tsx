"use client";

import { Moon, Sun } from "lucide-react";
import { useEffect, useState } from "react";

import { cn } from "@/lib/utils";

export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    const t = (document.documentElement.getAttribute("data-theme") as
      | "light"
      | "dark") ?? "light";
    setTheme(t);
  }, []);

  function toggle() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("rt-theme", next);
    } catch {
      // ignore storage failures (private mode, etc.)
    }
    setTheme(next);
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label="Toggle theme"
      className={cn(
        "inline-flex h-8 w-8 items-center justify-center rounded-full border border-[var(--color-rt-line)] text-[var(--color-rt-muted)] transition-colors hover:bg-[var(--color-rt-mist)] hover:text-[var(--color-rt-ink)]",
        className,
      )}
    >
      {theme === "dark" ? <Moon size={14} /> : <Sun size={14} />}
    </button>
  );
}
