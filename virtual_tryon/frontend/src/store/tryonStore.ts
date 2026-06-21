import { create } from "zustand";

export type Category = "upper_body" | "lower_body" | "dress" | "full_outfit";

export type TryOnResult = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  result_url?: string | null;
  error?: string | null;
  seed?: number | null;
  debug?: {
    mask_url?: string | null;
    agnostic_url?: string | null;
    core_output_url?: string | null;
    refined_output_url?: string | null;
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
  useRefiner: boolean;
  repairMode: boolean;
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
  useRefiner: true,
  repairMode: true,
  showDebug: true,
  loading: false,
  setField: (key, value) => set({ [key]: value } as Partial<TryOnState>),
  resetResult: () => set({ result: undefined, error: undefined, jobId: undefined })
}));
