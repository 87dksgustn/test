# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata
from pathlib import Path

datas = [('outputs', 'outputs'), ('outputs/latest_surrogate_bundle.pkl', 'embedded_bundle')]
_spec_file = globals().get('__file__')
if _spec_file:
    _spec_root = Path(_spec_file).resolve().parent
else:
    _spec_root = Path.cwd().resolve()
_bundle_sources = {
    "LFP_Gen2B": _spec_root / "outputs" / "latest_surrogate_bundle.pkl",
    "LFP_Gen2A": _spec_root.parent / "active_sampling_gp_mlp_optuna_v3_LFP_Gen2A" / "outputs" / "latest_surrogate_bundle.pkl",
    "MidNi_Gen1": _spec_root.parent / "active_sampling_gp_mlp_optuna_v3_MidNi_Gen1" / "outputs" / "latest_surrogate_bundle.pkl",
}
for _tag, _src in _bundle_sources.items():
    if _src.exists():
        datas.append((str(_src), f'embedded_bundle/{_tag}'))
binaries = []
hiddenimports = ['streamlit_surrogate_app', 'surrogate_bundle', 'config', 'acquisition', 'batch_selector', 'candidate_generator', 'data_loader', 'diagnostics', 'discrete_space', 'evaluation', 'metrics_utils', 'model_selector', 'models_gp', 'models_mlp', 'optuna_tuning', 'preprocessing']
datas += copy_metadata('streamlit')
tmp_ret = collect_all('streamlit')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['launcher_streamlit.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tensorflow', 'tensorboard', 'jax', 'jaxlib'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='SurrogatePredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
