import { create } from "zustand";

export type Category = "upper_body" | "lower_body" | "dress" | "full_outfit";
export type EngineMode = "" | "idm_vton" | "idm_vton_flux" | "idm_mask_expanded" | "idm_mask_expanded_flux" | "klein_lora" | "catvton";
export type PromptVariant = "default" | "strong_remove_old_garment" | "identity_strict";

export type ArtifactManifest = {
  job_id: string;
  files: {
    name: string;
    url: string;
    type: "image" | "json" | "csv" | "html" | "text";
    size_bytes: number;
  }[];
};

export type TryOnResult = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled" | "cancel_requested";
  result_url?: string | null;
  error?: string | null;
  error_code?: string | null;
  seed?: number | null;
  engine_status?: Record<string, string>;
  artifact_manifest?: ArtifactManifest | null;
  debug?: {
    mask_url?: string | null;
    mask_urls?: string[];
    agnostic_url?: string | null;
    core_output_url?: string | null;
    refined_output_url?: string | null;
    quality_report_url?: string | null;
    refine_mask_url?: string | null;
    prompt_core_url?: string | null;
    prompt_refine_url?: string | null;
    prompt_metadata_url?: string | null;
  };
  quality?: {
    needs_refine: boolean;
    notes: string[];
    background_preservation_score?: number | null;
    garment_similarity_score?: number | null;
    artifact_score?: number | null;
  } | null;
};

type TryOnState = {
  personImage?: File;
  topImage?: File;
  bottomImage?: File;
  dressImage?: File;
  category: Category;
  prompt: string;
  autoPrompt: boolean;
  testcaseId: string;
  promptVariant: PromptVariant;
  engineMode: EngineMode;
  useRefiner: boolean;
  repairMode: boolean;
  runMode: "sync" | "async";
  showDebug: boolean;
  loading: boolean;
  jobId?: string;
  result?: TryOnResult;
  error?: string;
  setField: <K extends keyof TryOnState>(key: K, value: TryOnState[K]) => void;
  resetResult: () => void;
};

export const useTryOnStore = create<TryOnState>((set) => ({
  category: "upper_body",
  prompt: "",
  autoPrompt: false,
  testcaseId: "",
  promptVariant: "default",
  engineMode: "",
  useRefiner: true,
  repairMode: true,
  runMode: "sync",
  showDebug: true,
  loading: false,
  setField: (key, value) => set({ [key]: value } as Partial<TryOnState>),
  resetResult: () => set({ result: undefined, error: undefined, jobId: undefined })
}));
