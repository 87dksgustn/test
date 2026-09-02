# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('outputs/latest_surrogate_bundle.pkl', 'embedded_bundle')]
binaries = []
hiddenimports = ['streamlit_surrogate_app', 'surrogate_bundle', 'config', 'acquisition', 'batch_selector', 'candidate_generator', 'config', 'data_loader', 'diagnostics', 'discrete_space', 'evaluation', 'metrics_utils', 'model_selector', 'models_gp', 'models_mlp', 'optuna_tuning', 'preprocessing', 'surrogate_bundle']
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
