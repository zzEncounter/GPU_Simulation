from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pybind11
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


ROOT = Path(__file__).parent.resolve()


def find_cuda_home() -> Path:
    # First check CUDA_HOME environment variable
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        return Path(cuda_home)
    
    # Check if nvcc is in conda environment
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_path = Path(nvcc).resolve()
        # Handle conda environments with targets/x86_64-linux structure
        if "targets" in nvcc_path.parts:
            idx = nvcc_path.parts.index("targets")
            return Path(*nvcc_path.parts[:idx])
        return nvcc_path.parent.parent
    
    # Check common conda CUDA locations
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        cuda_path = Path(conda_prefix) / "pkgs" / "cuda-nvcc-tools-13.2.78-he02047a_0"
        if cuda_path.exists():
            return cuda_path.parent.parent
        # Try to find any cuda-nvcc package
        for pkg in Path(conda_prefix / "pkgs").glob("cuda-nvcc-*"):
            if (pkg / "bin" / "nvcc").exists():
                return pkg.parent
    
    raise RuntimeError("nvcc was not found on PATH and CUDA_HOME is not set.")


class BuildCudaExtension(build_ext):
    def build_extension(self, ext: Extension) -> None:
        if not any(source.endswith(".cu") for source in ext.sources):
            super().build_extension(ext)
            return

        cuda_home = find_cuda_home()
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        ext_path = Path(self.get_ext_fullpath(ext.name))
        ext_path.parent.mkdir(parents=True, exist_ok=True)

        cxx_sources = [source for source in ext.sources if source.endswith(".cpp")]
        cuda_sources = [source for source in ext.sources if source.endswith(".cu")]

        extra_compile_args = getattr(ext, "extra_compile_args", {})
        cxx_args = list(extra_compile_args.get("cxx", []))
        nvcc_args = list(extra_compile_args.get("nvcc", []))

        objects = []
        if cxx_sources:
            objects.extend(
                self.compiler.compile(
                    cxx_sources,
                    output_dir=str(build_temp),
                    include_dirs=ext.include_dirs,
                    extra_postargs=cxx_args,
                    depends=ext.depends,
                )
            )

        for source in cuda_sources:
            source_path = Path(source)
            object_path = build_temp / f"{source_path.stem}.cu.o"
            cmd = [
                str(cuda_home / "bin" / "nvcc"),
                "-c",
                str(source_path),
                "-o",
                str(object_path),
                "-std=c++17",
                "-Xcompiler=-fPIC",
            ]
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
        )


def get_cuda_lib_path(cuda_home: Path) -> Path:
    """Get the CUDA library path, handling both standard and conda target structures."""
    # Standard CUDA: cuda_home/lib64
    lib64_path = cuda_home / "lib64"
    if lib64_path.exists():
        return lib64_path
    
    # Conda target structure: cuda_home/targets/x86_64-linux/lib
    target_lib_path = cuda_home / "targets" / "x86_64-linux" / "lib"
    if target_lib_path.exists():
        return target_lib_path
    
    # Fallback to lib64
    return lib64_path


ext_modules = [
    Extension(
        "standalone_backend._cuda_backend",
        sources=[
            "cpp/ising_cuda_bindings.cpp",
            "cpp/ising_cuda_backend.cu",
            "cpp/ising_cuda_kernels.cu",
        ],
        include_dirs=[
            str(ROOT / "cpp"),
            pybind11.get_include(),
            np.get_include(),
            str(find_cuda_home() / "include"),
        ],
        language="c++",
        libraries=["cudart", "cublas"],
        library_dirs=[str(get_cuda_lib_path(find_cuda_home()))],
        runtime_library_dirs=[str(get_cuda_lib_path(find_cuda_home()))],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17", "-fvisibility=hidden"],
            "nvcc": ["-O3", "--use_fast_math", "--extended-lambda", "-lineinfo"],
        },
    )
]


setup(
    name="standalone-ring-ising-backend",
    version="0.1.0",
    description="Minimal standalone CUDA backend for the ring Ising adjoint workflow.",
    packages=["standalone_backend"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildCudaExtension},
    zip_safe=False,
)
