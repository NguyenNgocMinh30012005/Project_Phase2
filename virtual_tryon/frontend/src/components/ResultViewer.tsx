import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { fetchJsonArtifact, resolveAssetUrl } from "../lib/api";
import { useTryOnStore } from "../store/tryonStore";

export function ResultViewer() {
  const result = useTryOnStore((state) => state.result);
  const showDebug = useTryOnStore((state) => state.showDebug);
  const [qualityReport, setQualityReport] = useState<unknown>();

  useEffect(() => {
    setQualityReport(undefined);
    if (!result?.debug?.quality_report_url) return;
    fetchJsonArtifact<unknown>(result.debug.quality_report_url)
      .then(setQualityReport)
      .catch(() => setQualityReport(undefined));
  }, [result?.debug?.quality_report_url]);

  if (!result) {
    return <section className="result-surface empty-state">No result yet</section>;
  }

  const resultUrl = resolveAssetUrl(result.result_url);
  const debugItems: [string, string | null | undefined][] = [
    ["Mask", result.debug?.mask_url],
    ["Refine mask", result.debug?.refine_mask_url],
    ["Agnostic", result.debug?.agnostic_url],
    ["Core", result.debug?.core_output_url],
    ["Refined", result.debug?.refined_output_url]
  ];

  return (
    <section className="result-surface">
      <div className="result-header">
        <div>
          <strong>{result.status}</strong>
          <span>{result.job_id}</span>
        </div>
        {resultUrl && (
          <a className="icon-button" href={resultUrl} download title="Download result">
            <Download size={18} />
          </a>
        )}
      </div>

      {result.error && <div className="error-box">{result.error}</div>}
      {resultUrl && <img className="result-image" src={resultUrl} alt="Try-on result" />}

      {showDebug && (
        <>
          <div className="debug-grid">
            {debugItems.map(([label, url]) => {
              const resolved = resolveAssetUrl(url);
              return (
                <figure key={label}>
                  {resolved ? <img src={resolved} alt={label} /> : <div className="empty-preview" />}
                  <figcaption>{label}</figcaption>
                </figure>
              );
            })}
            {(result.debug?.mask_urls ?? []).map((url, index) => {
              const resolved = resolveAssetUrl(url);
              return (
                <figure key={`${url}-${index}`}>
                  {resolved ? <img src={resolved} alt={`Mask ${index + 1}`} /> : <div className="empty-preview" />}
                  <figcaption>Mask {index + 1}</figcaption>
                </figure>
              );
            })}
          </div>
          {result.debug?.quality_report_url && (
            <a className="artifact-link" href={resolveAssetUrl(result.debug.quality_report_url)} target="_blank" rel="noreferrer">
              quality_report.json
            </a>
          )}
          {qualityReport ? <pre className="quality-json">{JSON.stringify(qualityReport, null, 2)}</pre> : null}
        </>
      )}

      {result.quality?.notes?.length ? (
        <div className="notes">
          {result.quality.notes.map((note) => <span key={note}>{note}</span>)}
        </div>
      ) : null}
    </section>
  );
}
