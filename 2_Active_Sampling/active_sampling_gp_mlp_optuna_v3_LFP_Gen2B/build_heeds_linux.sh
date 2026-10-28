#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "${1:-}" != "--inside-container" ]]; then
    if ! command -v docker >/dev/null; then
        printf '%s\n' 'Docker with Linux containers is required on the build machine.' >&2
        exit 1
    fi
    image="${HEEDS_BUILD_IMAGE:-quay.io/pypa/manylinux2014_x86_64:latest}"
    docker run --rm --platform linux/amd64 \
        --mount "type=bind,source=$PWD,target=/work" \
        -e "HEEDS_BUILD_IMAGE=$image" -w /work \
        "$image" bash /work/build_heeds_linux.sh --inside-container
    exit 0
fi

if [[ "$(uname -m)" != "x86_64" || "$(getconf GNU_LIBC_VERSION)" != "glibc 2.17" ]]; then
    printf '%s\n' 'Build requires x86_64 and glibc 2.17; use the manylinux2014 image.' >&2
    exit 1
fi

build_root="$(mktemp -d)"
trap 'rm -rf -- "$build_root"' EXIT
/opt/python/cp311-cp311/bin/python -m venv "$build_root/venv"
python="$build_root/venv/bin/python"
export MPLBACKEND=Agg
export MPLCONFIGDIR="$build_root/matplotlib"
export PYTHONNOUSERSITE=1

"$python" -m pip install --disable-pip-version-check --only-binary=:all: -r requirements_heeds_linux.txt
"$python" -m pip check
"$python" -m unittest test_heeds_launcher -v
"$python" launcher_heeds.py --check-runtime
"$python" -m PyInstaller --noconfirm --clean \
    --distpath dist_heeds_linux --workpath "$build_root/pyinstaller" HEEDSActiveSampler.spec

bundle="$PWD/dist_heeds_linux/heeds_active_sampler"
env -u LD_LIBRARY_PATH "$bundle/heeds_active_sampler" --check-runtime > "$bundle/runtime_check.json"
"$python" -m pip freeze > "$bundle/build_requirements.txt"
printf '%s\n' "${HEEDS_BUILD_IMAGE:-unknown}" > "$bundle/build_image.txt"
cp README_HEEDS_LINUX.md "$bundle/"
cp config.py "$bundle/config_snapshot.py.txt"

cd dist_heeds_linux
archive=heeds_active_sampler-linux-x86_64-glibc217.tar.gz
tar -czf "$archive" heeds_active_sampler
sha256sum "$archive" > "$archive.sha256"
printf 'Built: %s/%s\n' "$PWD" "$archive"