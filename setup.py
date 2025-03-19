from setuptools import setup
from Cython.build import cythonize
from rich.print import print as cprint
import typer

cprint("[bold green]Building Cython extension...[/bold green]")
setup()