import { Shirt, Upload } from "lucide-react";
import { Category, useTryOnStore } from "../store/tryonStore";

const categories: { value: Category; label: string }[] = [
  { value: "upper_body", label: "Top" },
  { value: "lower_body", label: "Bottom" },
  { value: "dress", label: "Dress" },
  { value: "full_outfit", label: "Full outfit" }
];

export function UploadGarment() {
  const state = useTryOnStore();
  const setField = state.setField;

  return (
    <section className="garment-section">
      <div className="segmented">
        {categories.map((item) => (
          <button
            type="button"
            key={item.value}
            className={state.category === item.value ? "active" : ""}
            onClick={() => setField("category", item.value)}
          >
            {item.label}
          </button>
        ))}
      </div>

      <div className="garment-grid">
        <label className="upload-panel">
          <span className="upload-icon"><Shirt size={18} /></span>
          <span className="upload-title">Top</span>
          <span className="upload-file">{state.topImage?.name ?? "No file selected"}</span>
          <input type="file" aria-label="Top garment image" accept="image/png,image/jpeg,image/webp" onChange={(e) => setField("topImage", e.target.files?.[0])} />
        </label>
        <label className="upload-panel">
          <span className="upload-icon"><Upload size={18} /></span>
          <span className="upload-title">Bottom</span>
          <span className="upload-file">{state.bottomImage?.name ?? "No file selected"}</span>
          <input type="file" aria-label="Bottom garment image" accept="image/png,image/jpeg,image/webp" onChange={(e) => setField("bottomImage", e.target.files?.[0])} />
        </label>
        <label className="upload-panel">
          <span className="upload-icon"><Upload size={18} /></span>
          <span className="upload-title">Dress</span>
          <span className="upload-file">{state.dressImage?.name ?? "No file selected"}</span>
          <input type="file" aria-label="Dress garment image" accept="image/png,image/jpeg,image/webp" onChange={(e) => setField("dressImage", e.target.files?.[0])} />
        </label>
      </div>
    </section>
  );
}
