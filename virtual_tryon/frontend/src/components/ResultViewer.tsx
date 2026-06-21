import { Download } from "lucide-react";
import { resolveAssetUrl } from "../lib/api";
import { useTryOnStore } from "../store/tryonStore";

export function ResultViewer() {
  const result = useTryOnStore((state) => state.result);
  const showDebug = useTryOnStore((state) => state.showDebug);

  if (!result) {
    return <section className="result-surface empty-state">No result yet</section>;
  }

  const resultUrl = resolveAssetUrl(result.result_url);
  const debugItems = [
    ["Mask", result.debug?.mask_url],
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
        </div>
      )}

      {result.quality?.notes?.length ? (
        <div className="notes">
          {result.quality.notes.map((note) => <span key={note}>{note}</span>)}
        </div>
      ) : null}
    </section>
  );
}
