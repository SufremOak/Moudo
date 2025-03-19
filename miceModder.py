import typer
import os
import subprocess

# from MouseLib import Mouse
# from MouseLib import MiceLib

app = typer.Typer()

@app.command()
def init(ProjectName: str):
    # Create config directory
    os.makedirs(f"{ProjectName}config/", exist_ok=True)
    
    # Create and write example mice.py file
    with open(f"{ProjectName}config/mice.py", 'w') as f:
        f.write('''from MoudoLib import Mouse

# Example mouse configuration
mouse = Mouse()
mouse.setDPI(800)
mouse.setPollingRate(1000)
mouse.setLiftOffDistance(2)
mouse.setAngleSnapping(False)

# Example button mapping
mouse.mapButton(1, "LEFT_CLICK")
mouse.mapButton(2, "RIGHT_CLICK")
mouse.mapButton(3, "MIDDLE_CLICK")
''')
    
    # Create and write example moudoProj.json
    with open(f"{ProjectName}config/moudoProj.json", 'w') as f:
        f.write('''{
    "name": "MyMouseConfig",
    "version": "1.0",
    "description": "Custom mouse configuration",
    "target": "mice.py",
    "output": "mod.json"
}''')
    
    # Create and write to BUILD file
    build_path = f"{ProjectName}config/BUILD"
    with open(build_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("mouc -f -Proj ./moudoProj.json\n")
        f.write("mkdir -p build\n")
        f.write("cd build\n")
        f.write("cp ../mod.json .\n")
        f.write("rm ../mod.json\n")
    
    # Make BUILD file executable
    os.chmod(build_path, 0o755)

@app.command()
def apply_mod(jsonPath: str):
    subprocess.run(["sudo", "cp", jsonPath, "/usr/local/moudo/config/mice(keppni).json"])

@app.command()
def build(ProjectName: str):
    build_script = f"{ProjectName}config/BUILD"
    if os.path.exists(build_script):
        subprocess.run(["bash", build_script])
    else:
        typer.echo(f"Build script not found at {build_script}")

if __name__ == "__main__":
    app()
