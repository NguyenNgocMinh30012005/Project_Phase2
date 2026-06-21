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

export function resolveAssetUrl(url?: string | null): string | undefined {
  if (!url) return undefined;
  if (url.startsWith("http")) return url;
  return `${API_BASE_URL}${url}`;
}
