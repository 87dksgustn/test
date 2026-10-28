# Surrogate Predictor Portable Deployment (Windows)

## Goal
- No-install executable package (ZIP)
- Embedded default model bundle
- Streamlit runs on localhost only
- Prediction history save is optional (default ON)
- Minimize visible source files in distributed package

## Build
1. Ensure the model bundles exist in the sibling projects:
	- Gen2B: `../active_sampling_gp_mlp_optuna_v3_LFP_Gen2B_SC40/outputs/latest_surrogate_bundle.pkl`
	- Gen2A: `../active_sampling_gp_mlp_optuna_v3_LFP_Gen2A/outputs/latest_surrogate_bundle.pkl`
	- Mid-Ni: `../active_sampling_gp_mlp_optuna_v3_MidNi_Gen1/outputs/latest_surrogate_bundle.pkl`
2. Run in PowerShell:

```powershell
./build_portable_windows.ps1 -AppVersion 2026.3 -OneFile
```

3. Output:
- `dist/SP_v2026.3.zip`

The script uses `SurrogatePredictor.spec` for both onefile and folder builds.
Omit `-OneFile` for a folder build; use `-Console` to show the console.
Runtime-only collection is always enabled, including when `-Lean` is omitted.

The existing direct onefile build command is unchanged:

```powershell
pyinstaller --noconfirm --clean .\SurrogatePredictor.spec
```

For a direct folder build, append `-- --onedir`. Add `--windowed` after `--`
to hide the console. Folder builds must be distributed with their full contents.

The build excludes the project's `outputs` tree, duplicate model bundle copies,
Streamlit `.agents`, `testing`, and `development` resources and modules.
Streamlit runtime source, web assets, metadata, and the model libraries are retained.
The `streamlit.hello` package must also be retained: Streamlit imports it during
normal server startup, even when this application is not running a demo.

## Code Signing (optional but recommended)

```powershell
./build_portable_windows.ps1 -AppVersion 2026.3 -OneFile -Sign -CertPath "C:\certs\company-code-signing.pfx"
```

Notes:
- Code signing proves publisher identity and reduces SmartScreen warnings.
- Timestamp keeps signature valid after certificate expiration.

## Run (on target PC)
1. Unzip package to any user-writable folder.
2. Run `SurrogatePredictor.exe`.
3. Browser opens local app URL (localhost).

## Security defaults
- Server bind: `127.0.0.1`
- No external hosting required
- Usage stats disabled

## Model strategy
- Embedded bundles: one per chemistry under `embedded_bundle/<tag>/latest_surrogate_bundle.pkl`
- Tags: `LFP_Gen2B`, `LFP_Gen2A`, `MidNi_Gen1`; Gen2B uses the SC40 source.
- Optional: user can override bundle path in UI

## History option
- Default ON
- User can turn OFF/ON in sidebar `Save prediction history`
- App chooses a writable path automatically and displays the active file location in the UI

## Operational cautions
- Keep package folder structure unchanged.
- Avoid protected folders requiring admin write permissions.
- For enterprise rollout, maintain versioned ZIP releases quarterly.
