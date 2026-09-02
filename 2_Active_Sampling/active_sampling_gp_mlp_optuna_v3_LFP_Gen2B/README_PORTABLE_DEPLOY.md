# Surrogate Predictor Portable Deployment (Windows)

## Goal
- No-install executable package (ZIP)
- Embedded default model bundle
- Streamlit runs on localhost only
- Prediction history save is optional (default ON)
- Minimize visible source files in distributed package

## Build
1. Ensure bundle exists at `outputs/latest_surrogate_bundle.pkl`.
2. Run in PowerShell:

```powershell
./build_portable_windows.ps1 -AppVersion 2026.3 -OneFile
```

3. Output:
- `dist/SP_v2026.3.zip`

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
- Default: embedded bundle (`embedded_bundle/latest_surrogate_bundle.pkl`)
- Optional: user can override bundle path in UI

## History option
- Default ON
- User can turn OFF/ON in sidebar `Save prediction history`
- App chooses a writable path automatically and displays the active file location in the UI

## Operational cautions
- Keep package folder structure unchanged.
- Avoid protected folders requiring admin write permissions.
- For enterprise rollout, maintain versioned ZIP releases quarterly.
