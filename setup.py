from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pybind11
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).resolve().parent
DEFAULT_CPP_DIR = ROOT / "cpp"
CPP_DIR_OVERRIDE_ENV = "CUDA_BACKEND_CPP_DIR"


def _resolve_repo_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def find_cpp_dir() -> Path:
    override = os.environ.get(CPP_DIR_OVERRIDE_ENV)
    if override:
        candidate = _resolve_repo_path(override)
        if candidate.is_dir():
            return candidate
        raise RuntimeError(
            f"{CPP_DIR_OVERRIDE_ENV} points to a missing directory: {candidate}"
        )

    if DEFAULT_CPP_DIR.is_dir():
        return DEFAULT_CPP_DIR

    matches = [
        path.parent
        for path in ROOT.rglob("ising_cuda_bindings.cpp")
        if "build" not in path.parts and ".venv" not in path.parts
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple candidate source directories found. Set {CPP_DIR_OVERRIDE_ENV}."
        )
    raise RuntimeError(
        f"Could not locate the CUDA source directory. Set {CPP_DIR_OVERRIDE_ENV}."
    )


def list_extension_sources(cpp_dir: Path) -> list[str]:
    sources = sorted(cpp_dir.glob("*.cpp")) + sorted(cpp_dir.glob("*.cu"))
    if not sources:
        raise RuntimeError(f"No extension sources found under {cpp_dir}")
    return [str(path.relative_to(ROOT)) for path in sources]


CPP_DIR = find_cpp_dir()


def _find_cuda_target_dir(cuda_home: Path) -> Path | None:
    targets_dir = cuda_home / "targets"
    if not targets_dir.is_dir():
        return None

    for target_dir in sorted(path for path in targets_dir.iterdir() if path.is_dir()):
        if (target_dir / "include").is_dir() or (target_dir / "lib").is_dir():
            return target_dir
    return None


def get_cuda_include_path(cuda_home: Path) -> Path:
    include_path = cuda_home / "include"
    if include_path.is_dir():
        return include_path

    target_dir = _find_cuda_target_dir(cuda_home)
    if target_dir is not None and (target_dir / "include").is_dir():
        return target_dir / "include"

    return include_path


def get_cuda_lib_path(cuda_home: Path) -> Path:
    for candidate in (cuda_home / "lib64", cuda_home / "lib"):
        if candidate.is_dir():
            return candidate

    target_dir = _find_cuda_target_dir(cuda_home)
    if target_dir is not None:
        for candidate in (target_dir / "lib64", target_dir / "lib"):
            if candidate.is_dir():
                return candidate

    return cuda_home / "lib64"


def _candidate_cuda_homes() -> list[Path]:
    candidates: list[Path] = []

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(env_var)
        if value:
            candidates.append(Path(value).expanduser())

    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_path = Path(nvcc).resolve()
        if nvcc_path.parent.name == "bin":
            candidates.append(nvcc_path.parent.parent)
        if "targets" in nvcc_path.parts:
            idx = nvcc_path.parts.index("targets")
            candidates.append(Path(*nvcc_path.parts[:idx]))

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix).expanduser())

    return candidates


def find_cuda_home() -> Path:
    checked: list[str] = []
    seen: set[Path] = set()

    for candidate in _candidate_cuda_homes():
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            checked.append(str(candidate))
            continue

        if resolved in seen:
            continue
        seen.add(resolved)
        checked.append(str(resolved))

        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        nvcc_path = resolved / "bin" / nvcc_name
        include_path = get_cuda_include_path(resolved)
        lib_path = get_cuda_lib_path(resolved)
        if nvcc_path.exists() and include_path.is_dir() and lib_path.is_dir():
            return resolved

    searched = ", ".join(checked) if checked else "no candidates"
    raise RuntimeError(
        "Could not locate a usable CUDA toolkit. Set CUDA_HOME/CUDA_PATH or put nvcc on "
        f"PATH. Checked: {searched}"
    )


def _unique_path_strings(paths: list[str | Path]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for path_like in paths:
        path = str(path_like)
        if path and path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def _split_compile_args(ext: Extension) -> tuple[list[str], list[str]]:
    extra_compile_args = getattr(ext, "extra_compile_args", {})
    if isinstance(extra_compile_args, dict):
        return (
            list(extra_compile_args.get("cxx", [])),
            list(extra_compile_args.get("nvcc", [])),
        )
    return (list(extra_compile_args), [])


def _resolve_source_path(source: str) -> Path:
    return _resolve_repo_path(source)


def _cuda_object_path(build_temp: Path, source_path: Path) -> Path:
    relative = source_path.relative_to(ROOT)
    safe_name = "_".join(relative.parts).replace(".", "_")
    return build_temp / f"{safe_name}.o"


def default_cxx_compile_args() -> list[str]:
    if os.name == "nt":
        return ["/O2", "/std:c++17"]
    return ["-O3", "-std=c++17", "-fvisibility=hidden"]


def default_nvcc_compile_args() -> list[str]:
    return ["-O3", "--use_fast_math", "--extended-lambda", "-lineinfo"]


class BuildCudaExtension(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not any(source.endswith(".cu") for source in ext.sources):
            super().build_extension(ext)
            return

        cuda_home = find_cuda_home()
        cuda_include_dir = get_cuda_include_path(cuda_home)
        cuda_lib_dir = get_cuda_lib_path(cuda_home)

        ext.include_dirs = _unique_path_strings(
            [
                *(ext.include_dirs or []),
                CPP_DIR,
                pybind11.get_include(),
                np.get_include(),
                cuda_include_dir,
            ]
        )
        ext.library_dirs = _unique_path_strings([*(ext.library_dirs or []), cuda_lib_dir])
        if os.name != "nt":
            ext.runtime_library_dirs = _unique_path_strings(
                [*(ext.runtime_library_dirs or []), cuda_lib_dir]
            )

        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        ext_path = Path(self.get_ext_fullpath(ext.name))
        ext_path.parent.mkdir(parents=True, exist_ok=True)

        cxx_sources = [_resolve_source_path(source) for source in ext.sources if source.endswith(".cpp")]
        cuda_sources = [_resolve_source_path(source) for source in ext.sources if source.endswith(".cu")]

        cxx_args, nvcc_args = _split_compile_args(ext)

        objects: list[str] = []
        if cxx_sources:
            objects.extend(
                self.compiler.compile(
                    [str(source) for source in cxx_sources],
                    output_dir=str(build_temp),
                    include_dirs=ext.include_dirs,
                    extra_postargs=cxx_args,
                    depends=ext.depends,
                )
            )

        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        nvcc_path = cuda_home / "bin" / nvcc_name

        for source_path in cuda_sources:
            object_path = _cuda_object_path(build_temp, source_path)
            cmd = [
                str(nvcc_path),
                "-c",
                str(source_path),
                "-o",
                str(object_path),
                "-std=c++17",
            ]
            if os.name != "nt":
                cmd.append("-Xcompiler=-fPIC")
            for include_dir in ext.include_dirs:
                cmd.extend(["-I", str(include_dir)])
            cmd.extend(nvcc_args)
            subprocess.check_call(cmd, cwd=str(ROOT))
            objects.append(str(object_path))

        link_args = list(getattr(ext, "extra_link_args", []))
        self.compiler.link_shared_object(
            objects,
            str(ext_path),
            libraries=ext.libraries,
            library_dirs=ext.library_dirs,
            runtime_library_dirs=ext.runtime_library_dirs,
            extra_postargs=link_args,
            target_lang=ext.language,
        )


ext_modules = [
    Extension(
        "standalone_backend._cuda_backend",
        sources=list_extension_sources(CPP_DIR),
        include_dirs=[str(CPP_DIR)],
        language="c++",
        libraries=["cudart", "cublas"],
        library_dirs=[],
        runtime_library_dirs=[],
        extra_compile_args={
            "cxx": default_cxx_compile_args(),
            "nvcc": default_nvcc_compile_args(),
        },
    )
]


setup(
    name="standalone-ring-ising-backend",
    version="0.1.0",
    description="Minimal standalone CUDA backend for the ring Ising adjoint workflow.",
    packages=find_packages(
        include=[
            "ring_ising",
            "ring_ising.*",
            "standalone_backend",
            "standalone_backend.*",
        ]
    ),
    py_modules=[
        "run_pennylane_baseline",
        "run_standalone_backend",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCudaExtension},
    zip_safe=False,
)
