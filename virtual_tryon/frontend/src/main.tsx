import React from "react";
import ReactDOM from "react-dom/client";
import { Loader2, Play } from "lucide-react";
import { submitTryOn } from "./lib/api";
import { ResultViewer } from "./components/ResultViewer";
import { TryOnPreview } from "./components/TryOnPreview";
import { UploadGarment } from "./components/UploadGarment";
import { UploadPerson } from "./components/UploadPerson";
import { useTryOnStore } from "./store/tryonStore";
import "./styles.css";

function App() {
  const state = useTryOnStore();
  const setField = state.setField;

  async function generate() {
    state.resetResult();
    setField("loading", true);
    try {
      if (!state.personImage) throw new Error("Person image is required.");
      const form = new FormData();
      form.append("person_image", state.personImage);
      if (state.topImage) form.append("garment_top", state.topImage);
      if (state.bottomImage) form.append("garment_bottom", state.bottomImage);
      if (state.dressImage) form.append("garment_dress", state.dressImage);
      form.append("category", state.category);
      form.append("prompt", state.prompt);
      form.append("use_refiner", String(state.useRefiner));
      form.append("repair_mode", String(state.repairMode));
      const result = await submitTryOn(form);
      setField("result", result);
      setField("jobId", result.job_id);
      if (result.error) setField("error", result.error);
    } catch (error) {
      setField("error", error instanceof Error ? error.message : String(error));
    } finally {
      setField("loading", false);
    }
  }

  return (
    <main className="app-shell">
      <section className="workbench">
        <div className="toolbar">
          <div>
            <h1>Virtual Try-On</h1>
            <span>{state.jobId ?? "Ready"}</span>
          </div>
          <button className="primary-button" type="button" onClick={generate} disabled={state.loading}>
            {state.loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
            Generate
          </button>
        </div>

        <div className="input-grid">
          <UploadPerson />
          <UploadGarment />
        </div>

        <div className="control-row">
          <label className="prompt-box">
            <span>Prompt</span>
            <textarea value={state.prompt} onChange={(e) => setField("prompt", e.target.value)} />
          </label>
          <div className="switches">
            <label><input type="checkbox" checked={state.useRefiner} onChange={(e) => setField("useRefiner", e.target.checked)} /> FLUX refine</label>
            <label><input type="checkbox" checked={state.repairMode} onChange={(e) => setField("repairMode", e.target.checked)} /> Repair</label>
            <label><input type="checkbox" checked={state.showDebug} onChange={(e) => setField("showDebug", e.target.checked)} /> Debug</label>
          </div>
        </div>

        {state.error && <div className="error-box">{state.error}</div>}

        <div className="preview-grid">
          <TryOnPreview title="Person" file={state.personImage} />
          <TryOnPreview title="Top" file={state.topImage} />
          <TryOnPreview title="Bottom" file={state.bottomImage} />
          <TryOnPreview title="Dress" file={state.dressImage} />
        </div>
      </section>

      <ResultViewer />
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
