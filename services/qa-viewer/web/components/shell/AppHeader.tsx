"use client";

import { useQuery } from "@tanstack/react-query";
import { Cpu, Layers, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { LogoLockup } from "@/components/brand/Logo";
import { ThemeToggle } from "@/components/shell/ThemeToggle";
import { getQueue, type QueueEntry } from "@/lib/api";

export function AppHeader() {
  return (
    <header className="relative flex h-[68px] shrink-0 items-center justify-between border-b border-[var(--color-rt-line)] bg-[var(--color-rt-paper)]/85 px-5 backdrop-blur-md sm:px-6">
      {/* Hairline accent — a thin top rule in the brand gradient so the
          console reads as a deliberate instrument plate, not a plain bar. */}
      <div
        aria-hidden
        className="bg-rt-gradient pointer-events-none absolute inset-x-0 top-0 h-[2px] opacity-70"
      />
      <div className="flex items-center gap-3">
        <LogoLockup />
      </div>
      <div className="flex items-center gap-2">
        <NodeChip />
        <QueueWidget />
        <ThemeToggle />
      </div>
    </header>
  );
}

// Compact identity chip — this viewer owns one GPU on the cluster. Surfacing it
// gives the console a sense of "where am I" without another data dependency.
function NodeChip() {
  return (
    <span
      className="hidden items-center gap-1.5 rounded-full border border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] px-2.5 py-1 text-[10.5px] font-medium text-[var(--color-rt-muted)] md:inline-flex"
      title="QA viewer · pinned to a dedicated GPU"
    >
      <Cpu size={11} className="text-[var(--color-rt-accent)]" />
      GPU 0 · QA
    </span>
  );
}

// Polls /api/queue and renders a click-to-expand pill showing the global
// inference queue. Lets a second reviewer see "1 running, 1 queued"
// instead of silently waiting for the GPU.
function QueueWidget() {
  const { data } = useQuery({
    queryKey: ["predict-queue"],
    queryFn: getQueue,
    // Cheap (in-memory backend); 5 s is a fine cadence — the queue rarely
    // turns over faster than that for ResEnc-XL workloads.
    refetchInterval: 5000,
    staleTime: 4000,
  });
  const [open, setOpen] = useState(false);
  const wrapRef = useRef<HTMLDivElement | null>(null);

  // It's a disclosure, not a modal dialog — dismiss on Escape or an
  // outside click (mirrors the ModelCard color picker) so keyboard and
  // mouse users can both close it without re-clicking the trigger.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const depth = data?.depth ?? 0;
  const runningCount = data?.in_flight.filter((e) => e.state === "running").length ?? 0;
  const queuedCount = depth - runningCount;

  const tone =
    depth === 0
      ? "muted"
      : runningCount > 0
      ? "accent"
      : "muted";

  const label =
    depth === 0
      ? "GPU idle"
      : runningCount > 0 && queuedCount === 0
      ? `${runningCount} running`
      : queuedCount > 0
      ? `${runningCount} running · ${queuedCount} queued`
      : `${depth} active`;

  return (
    <div className="relative" ref={wrapRef}>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium tracking-wide transition-colors ${
          tone === "accent"
            ? "border-[color-mix(in_oklab,var(--color-rt-accent)_30%,transparent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-accent)]"
            : "border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] text-[var(--color-rt-muted)]"
        }`}
        aria-expanded={open}
        aria-label="Predict queue"
        title="Predict queue"
      >
        {runningCount > 0 ? <Loader2 size={12} className="animate-spin" /> : <Layers size={12} />}
        {label}
      </button>
      {open && (
        <div className="absolute right-0 z-40 mt-2 w-[360px] rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-paper)] p-3 shadow-[var(--shadow-rt-elevation-3)]">
          <div className="mb-2 flex items-center justify-between text-[11px] uppercase tracking-[0.16em] text-[var(--color-rt-muted)]">
            <span>Predict queue</span>
            <span>{depth} active</span>
          </div>
          {depth === 0 ? (
            <div className="py-3 text-center text-[12px] text-[var(--color-rt-muted)]">
              No inference running.
            </div>
          ) : (
            <ul className="flex flex-col gap-2">
              {data!.in_flight.map((e) => (
                <QueueRow key={e.prediction_id} entry={e} />
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}

function QueueRow({ entry }: { entry: QueueEntry }) {
  const datasetShort = entry.model_id.split("::")[0].replace(/^Dataset(\d+)_/, "D$1 ");
  const caseShort = entry.case_id.split("/")[1] ?? entry.case_id;
  const etaText = entry.eta_s
    ? `~${Math.round(entry.eta_s)}s`
    : "—";
  return (
    <li className="flex items-start gap-2 rounded-[var(--radius-rt-sm)] border border-[var(--color-rt-line)] bg-[var(--color-rt-mist)] px-2 py-1.5">
      <span
        className={`mt-0.5 inline-block h-2 w-2 shrink-0 rounded-full ${
          entry.state === "running"
            ? "bg-[var(--color-rt-accent)]"
            : "bg-[var(--color-rt-muted)]"
        }`}
        aria-hidden
      />
      <div className="min-w-0 flex-1">
        <div className="truncate text-[12px] font-medium text-[var(--color-rt-ink)]">
          {datasetShort}
        </div>
        <div className="truncate text-[10.5px] text-[var(--color-rt-muted)]">
          {caseShort} · {entry.state === "queued" ? `pos ${entry.position_in_queue}` : "running"} · eta {etaText}
        </div>
        {entry.reviewer && (
          <div className="truncate text-[10px] text-[var(--color-rt-muted)]">
            reviewer · {entry.reviewer}
          </div>
        )}
      </div>
    </li>
  );
}
