# -*- mode: python ; coding: utf-8 -*-
import argparse
from PyInstaller.utils.hooks import collect_all
from pathlib import Path

_parser = argparse.ArgumentParser()
_parser.add_argument('--onedir', action='store_true')
_parser.add_argument('--windowed', action='store_true')
_options = _parser.parse_args()

_spec_file = globals().get('__file__')
if _spec_file:
    _spec_root = Path(_spec_file).resolve().parent
else:
    _spec_root = Path.cwd().resolve()
_sc40_bundle = _spec_root.parent / "active_sampling_gp_mlp_optuna_v3_LFP_Gen2B_SC40" / "outputs" / "latest_surrogate_bundle.pkl"
if not _sc40_bundle.is_file():
    raise FileNotFoundError(f"SC40 bundle not found: {_sc40_bundle}")
datas = []
_bundle_sources = {
    "LFP_Gen2B": _sc40_bundle,
    "LFP_Gen2A": _spec_root.parent / "active_sampling_gp_mlp_optuna_v3_LFP_Gen2A" / "outputs" / "latest_surrogate_bundle.pkl",
    "MidNi_Gen1": _spec_root.parent / "active_sampling_gp_mlp_optuna_v3_MidNi_Gen1" / "outputs" / "latest_surrogate_bundle.pkl",
}
for _tag, _src in _bundle_sources.items():
    if _src.exists():
        datas.append((str(_src), f'embedded_bundle/{_tag}'))
binaries = []
hiddenimports = ['streamlit_surrogate_app', 'surrogate_bundle', 'config', 'acquisition', 'metrics_utils', 'models_gp', 'models_mlp', 'preprocessing']
_streamlit_excludes = ['streamlit.testing', 'streamlit.development']
tmp_ret = collect_all(
    'streamlit',
    exclude_datas=['.agents/**', 'testing/**', 'development/**'],
    filter_submodules=lambda name: not any(
        name == excluded or name.startswith(excluded + '.') for excluded in _streamlit_excludes
    ),
)
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
    excludes=['tensorflow', 'tensorboard', 'jax', 'jaxlib'] + _streamlit_excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [] if _options.onedir else a.binaries,
    [] if _options.onedir else a.datas,
    [],
    exclude_binaries=_options.onedir,
    name='SurrogatePredictor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=not _options.windowed,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

if _options.onedir:
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        name='SurrogatePredictor',
    )
