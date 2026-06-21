# Current Limitations

- Fine garment logos, typography, patterns, and small accessories can be distorted.
- Large pose changes and unusual camera angles reduce garment alignment quality.
- Occluded torso regions are difficult to reconstruct reliably.
- Long hair and hands overlapping the garment can cause mask or boundary artifacts.
- The optional refiner can over-edit identity, background, or garment details despite mask constraints.
- Runtime and output behavior depend on third-party checkpoint availability and model licenses.
- The demo currently has no authentication, tenant isolation, quotas, or per-user artifact access control.
- Cancellation is cooperative: an active GPU process finishes before the job becomes fully cancelled.
- Timeout enforcement marks an overlong attempt as failed after it returns; it does not terminate the model subprocess yet.
