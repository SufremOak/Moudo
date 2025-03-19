from setuptools import setup
from Cython.build import cythonize
from rich.print import print as cprint
import subprocess as sb

def check_compiler(name):
    try:
        sb.check_output([name, '--version'])
        cprint(f"[green]{name} is installed[/green]")
        return True
    except (sb.CalledProcessError, FileNotFoundError):
        cprint(f"[red]{name} not found[/red]")
        return False

if not check_compiler('g++'):
    raise SystemExit("g++ compiler required")

if not check_compiler('nvcc'):
    cprint("[yellow]CUDA not found - GPU acceleration will be disabled[/yellow]")

cprint("[bold green]Building Cython extension...[/bold green]")
setup(
    ext_modules = cythonize("./src/MouseLib.pyx")
)

cprint("[bold green]Cython Build complete![/bold green]")
cprint("[bold green]Building Cuda and RGB Controls package...[/bold green]")
sb.run("cd build && cmake .. && cd nv && make", shell=True)
cprint("[bold green]Build complete![/bold green]")
exit(0)