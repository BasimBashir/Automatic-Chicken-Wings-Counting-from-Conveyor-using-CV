# cython_build.py
#
# Compiles app/core/*.py and app/routers/*.py -> native .so inside the Docker
# builder stage so the shipped image contains no readable Python source for the
# core logic or the API routers.
#
# Runs ONLY during `docker build`. It never modifies local source files.
# Invoke with:  python cython_build.py build_ext --inplace
import glob
import os

from setuptools import Extension, setup
from Cython.Build import cythonize

COMPILE_DIRS = ["app/core", "app/routers"]

# Modules left as plain .py (by basename). `auth` kept for parity with the
# other counters (this repo currently has no auth module).
EXCLUDE = {"auth"}


def _modules():
    found = []
    for d in COMPILE_DIRS:
        for path in glob.glob(f"{d}/*.py"):
            base = os.path.splitext(os.path.basename(path))[0]
            if base == "__init__" or base in EXCLUDE:
                continue
            found.append((d.replace("/", ".") + "." + base, path))
    return sorted(found)


def main():
    mods = _modules()
    print(f"[cython] compiling {len(mods)} modules:", flush=True)
    for qual, _ in mods:
        print(f"  {qual}", flush=True)
    if not mods:
        return
    extensions = [Extension(qual, [path]) for qual, path in mods]
    setup(
        ext_modules=cythonize(
            extensions,
            nthreads=os.cpu_count() or 1,
            compiler_directives={
                "language_level": "3",
                "binding": True,           # full function objects -> signatures introspectable
                "embedsignature": True,    # keep signatures in docstrings
                "always_allow_keywords": True,
                # Keep annotations as hints only — do NOT let Cython treat
                # `x: float` / `x: int` as C types. False == faithful
                # pure-Python semantics.
                "annotation_typing": False,
                "profile": False,
                "linetrace": False,
            },
        ),
    )


if __name__ == "__main__":
    main()
