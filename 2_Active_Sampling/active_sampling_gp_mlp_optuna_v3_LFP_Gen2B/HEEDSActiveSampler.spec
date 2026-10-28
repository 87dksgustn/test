import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

if sys.platform != "linux":
    raise SystemExit("Use build_heeds_linux.sh in the Linux build container.")

root = Path(SPECPATH)
analysis = Analysis(
    [str(root / "launcher_heeds.py")],
    pathex=[str(root)],
    binaries=[],
    datas=copy_metadata("optuna"),
    hiddenimports=["0_main_active_sampling", *collect_submodules("optuna")],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["Agg"]}},
    excludes=["streamlit", "tkinter", "PyQt5", "PyQt6", "PySide2", "PySide6"],
    noarchive=False,
    module_collection_mode={"torch": "pyz+py"},
)
pyz = PYZ(analysis.pure)
executable = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name="heeds_active_sampler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
collection = COLLECT(
    executable, analysis.binaries, analysis.datas,
    strip=False, upx=False, name="heeds_active_sampler",
)