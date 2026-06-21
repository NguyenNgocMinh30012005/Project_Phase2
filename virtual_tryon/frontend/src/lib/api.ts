import type { TryOnResult } from "../store/tryonStore";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export async function submitTryOn(form: FormData): Promise<TryOnResult> {
  const response = await fetch(`${API_BASE_URL}/tryon`, {
    method: "POST",
    body: form
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Request failed with ${response.status}`);
  }
  return response.json();
}

export async function getTryOnJob(jobId: string): Promise<TryOnResult> {
  const response = await fetch(`${API_BASE_URL}/tryon/${jobId}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Status request failed with ${response.status}`);
  }
  return response.json();
}

export async function cancelTryOnJob(jobId: string): Promise<TryOnResult> {
  const response = await fetch(`${API_BASE_URL}/tryon/${jobId}`, { method: "DELETE" });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.detail ?? `Cancel request failed with ${response.status}`);
  }
  return response.json();
}

export async function fetchJsonArtifact<T>(url?: string | null): Promise<T | undefined> {
  const resolved = resolveAssetUrl(url);
  if (!resolved) return undefined;
  const response = await fetch(resolved);
  if (!response.ok) return undefined;
  return response.json();
}

export function resolveAssetUrl(url?: string | null): string | undefined {
  if (!url) return undefined;
  if (url.startsWith("http")) return url;
  return `${API_BASE_URL}${url}`;
}
