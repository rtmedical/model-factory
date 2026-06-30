"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  CheckCircle2,
  ImageOff,
  Loader2,
  PackagePlus,
  Plus,
  UploadCloud,
} from "lucide-react";
import { useState } from "react";

import { donateCase, getCohort, type CaseInfo } from "@/lib/api";
import { useQAStore } from "@/lib/store";
import { cn, regionLabel } from "@/lib/utils";

import { UploadCaseButton } from "./UploadCaseButton";

export function CaseStrip() {
  const { data } = useQuery({ queryKey: ["cohort"], queryFn: getCohort });
  const selectedModel = useQAStore((s) => s.selectedModel);
  const selectedCase = useQAStore((s) => s.selectedCase);
  const setCase = useQAStore((s) => s.setCase);

  const cases = data?.cases ?? [];
  // With a model selected, show only ITS compatible cases (donated cohort
  // cases + uploads made against it) so the strip is a real picker, not a
  // wall of disabled chips from other datasets sharing the region.
  const compatible = selectedModel
    ? cases.filter((c) => c.compatible_models.includes(selectedModel.model_id))
    : cases;

  if (!data) {
    return (
      <div className="flex h-16 items-center justify-center text-[12px] text-[var(--color-rt-muted)]">
        Loading cases…
      </div>
    );
  }

  // Model selected but nothing compatible yet — offer donate + upload so the
  // reviewer can get a case without dropping to the CLI.
  if (selectedModel && compatible.length === 0) {
    return <NoCasesRow modelId={selectedModel.model_id} />;
  }

  if (!compatible.length) {
    return (
      <div className="flex h-16 items-center justify-center gap-2 text-[12px] text-[var(--color-rt-muted)]">
        <ImageOff size={14} />
        No cohort cases yet — run{" "}
        <code className="rounded bg-[var(--color-rt-mist)] px-1 py-0.5 font-mono text-[11px]">
          modelfactory qa cohort prepare
        </code>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2 overflow-x-auto py-1 pl-1">
      {compatible.map((c) => (
        <CaseChip
          key={c.case_id}
          case={c}
          active={selectedCase?.case_id === c.case_id}
          onClick={() => setCase(c)}
        />
      ))}
      {selectedModel && (
        <>
          <AddCaseChip modelId={selectedModel.model_id} currentCount={compatible.length} />
          <UploadCaseButton modelId={selectedModel.model_id} />
        </>
      )}
    </div>
  );
}

function CaseChip({
  case: c,
  active,
  onClick,
}: {
  case: CaseInfo;
  active: boolean;
  onClick: () => void;
}) {
  const isUpload = c.uploaded || c.source_dataset === "uploaded";
  const hasGt = !!c.groundtruth_path;
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "shrink-0 rounded-[var(--radius-rt-sm)] border px-3 py-1.5 text-left transition-all",
        active
          ? "border-[var(--color-rt-accent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_12%,var(--color-rt-paper))] shadow-[inset_0_0_0_1px_var(--color-rt-accent)]"
          : "border-[var(--color-rt-line)] hover:border-[var(--color-rt-line-2)] hover:bg-[var(--color-rt-mist)]",
      )}
    >
      <div className="flex items-center gap-1.5 text-[12px] font-medium text-[var(--color-rt-ink)]">
        {active && (
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--color-rt-accent)]" />
        )}
        {c.case_id.split("/")[1]}
        {isUpload && (
          <UploadCloud size={11} className="text-[var(--color-rt-accent)]" />
        )}
      </div>
      <div className="text-[10px] text-[var(--color-rt-muted)]">
        {isUpload
          ? hasGt
            ? "uploaded · GT set"
            : "uploaded"
          : regionLabel(c.region)}
      </div>
    </button>
  );
}

// "+ case" chip — tops the dataset's cohort up by one via the additive
// donate endpoint. When the dataset is exhausted (already_existed), surfaces
// a quiet "no more cases" hint instead of silently doing nothing.
function AddCaseChip({
  modelId,
  currentCount,
}: {
  modelId: string;
  currentCount: number;
}) {
  const qc = useQueryClient();
  const setCase = useQAStore((s) => s.setCase);
  const [note, setNote] = useState<string | null>(null);

  const donate = useMutation({
    mutationFn: () => donateCase({ model_id: modelId, n_pick: currentCount + 1 }),
    onMutate: () => setNote(null),
    onSuccess: async (resp) => {
      await qc.invalidateQueries({ queryKey: ["cohort"] });
      if (!resp.already_existed && resp.new_cases.length > 0) {
        setCase(resp.new_cases[resp.new_cases.length - 1]);
      } else if (resp.already_existed) {
        setNote("no more cases in dataset");
      }
    },
    onError: (e: unknown) => setNote(e instanceof Error ? e.message : String(e)),
  });

  return (
    <div className="flex shrink-0 flex-col items-stretch">
      <button
        type="button"
        onClick={() => donate.mutate()}
        disabled={donate.isPending}
        title="Add another case from this dataset to the QA cohort"
        className={cn(
          "inline-flex items-center gap-1.5 rounded-[var(--radius-rt-sm)] border px-3 py-1.5 text-[12px] font-medium transition-colors",
          "border-[var(--color-rt-line)] text-[var(--color-rt-muted)] hover:border-[var(--color-rt-accent)] hover:text-[var(--color-rt-accent)]",
          donate.isPending && "cursor-not-allowed opacity-70",
        )}
      >
        {donate.isPending ? (
          <Loader2 size={12} className="animate-spin" />
        ) : (
          <Plus size={12} />
        )}
        case
      </button>
      {note && (
        <span
          className="mt-0.5 max-w-[160px] truncate text-[10px] text-[var(--color-rt-muted)]"
          title={note}
        >
          {note}
        </span>
      )}
    </div>
  );
}

// Empty-state for a model whose source dataset has no compatible case yet.
// Offers Donate (materialize one from the dataset's imagesTr) AND Upload
// (bring your own DICOM/NIfTI). Replaces the old dead-end "run CLI" message.
function NoCasesRow({ modelId }: { modelId: string }) {
  const qc = useQueryClient();
  const setCase = useQAStore((s) => s.setCase);
  const datasetShort = modelId.split("::")[0].replace(/^Dataset(\d+)_/, "D$1 ");
  const [errMsg, setErrMsg] = useState<string | null>(null);

  const donate = useMutation({
    mutationFn: () => donateCase({ model_id: modelId }),
    onMutate: () => setErrMsg(null),
    onSuccess: async (resp) => {
      await qc.invalidateQueries({ queryKey: ["cohort"] });
      if (resp.new_cases.length > 0) {
        setCase(resp.new_cases[0]);
      }
    },
    onError: (e: unknown) => {
      setErrMsg(e instanceof Error ? e.message : String(e));
    },
  });

  return (
    <div className="flex h-16 items-center justify-center gap-3 text-[12px] text-[var(--color-rt-muted)]">
      <ImageOff size={14} />
      <span>
        No QA case for <span className="font-medium">{datasetShort}</span> yet.
      </span>
      <button
        type="button"
        onClick={() => donate.mutate()}
        disabled={donate.isPending}
        className={cn(
          "inline-flex items-center gap-1.5 rounded-[var(--radius-rt-sm)] border px-2.5 py-1 font-medium",
          "border-[color-mix(in_oklab,var(--color-rt-accent)_30%,transparent)] bg-[color-mix(in_oklab,var(--color-rt-accent)_10%,var(--color-rt-paper))] text-[var(--color-rt-accent)]",
          "hover:bg-[color-mix(in_oklab,var(--color-rt-accent)_18%,var(--color-rt-paper))]",
          donate.isPending && "cursor-not-allowed opacity-60",
        )}
        title="Pick a case from this dataset's imagesTr and copy it into the cohort."
      >
        {donate.isPending ? (
          <Loader2 size={12} className="animate-spin" />
        ) : donate.isSuccess ? (
          <CheckCircle2 size={12} />
        ) : (
          <PackagePlus size={12} />
        )}
        {donate.isPending ? "Donating…" : donate.isSuccess ? "Donated" : "Donate a case"}
      </button>
      <span className="text-[var(--color-rt-muted)]">or</span>
      <UploadCaseButton modelId={modelId} />
      {errMsg && (
        <span className="text-[11px] text-[var(--color-rt-pip-error)]">{errMsg}</span>
      )}
    </div>
  );
}
