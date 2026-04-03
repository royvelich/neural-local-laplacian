import sys
import subprocess

print(f"Python version: {sys.version}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version (runtime): {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
except ImportError:
    print("PyTorch: not installed")

# Check NVIDIA driver / CUDA toolkit via nvidia-smi
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print(f"NVIDIA driver version: {result.stdout.strip()}")
except (FileNotFoundError, subprocess.TimeoutExpired):
    print("nvidia-smi: not found")

# Check cuDSS
# Check cuDSS: try Python module first, then search site-packages for the redistributable
cudss_found = False
for mod_name in ["nvidia.cudss", "cudss"]:
    try:
        import importlib
        mod = importlib.import_module(mod_name)
        ver = getattr(mod, "__version__", "unknown")
        print(f"cuDSS installed: yes (module: {mod_name}, version: {ver})")
        cudss_found = True
        break
    except ImportError:
        continue

if not cudss_found:
    # nvidia-cudss-cu12 is a redistributable: DLLs/SOs in site-packages, no Python binding
    import site
    from pathlib import Path

    lib_names = ["cudss64_0.dll", "cudss.dll", "libcudss.so", "libcudss.so.0"]
    for sp in site.getsitepackages():
        for match in Path(sp).rglob("cudss*"):
            if match.suffix in (".dll", ".so") or ".so." in match.name:
                # Try to get version from dist-info directory nearby
                ver = "unknown"
                for dist in Path(sp).glob("nvidia_cudss_cu*-*.dist-info"):
                    ver = dist.name.split("-")[1]  # e.g. "0.7.1.6"
                print(f"cuDSS installed: yes (native lib: {match}, version: {ver})")
                cudss_found = True
                break
        if cudss_found:
            break

if not cudss_found:
    print("cuDSS installed: no")