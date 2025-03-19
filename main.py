import typer
import subprocess

app = typer.Typer()

def check_compiler(name):
    try:
        subprocess.check_output([name, '--version'])
        print(f"{name} is installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{name} not found")
        return False

def cc():
    pass

@app.command()
def start():
    print("Welcome to the Moudo version {version} toolkit menu!")
    print("Type 'help' for a list of commands.")
    while True:
        command = input("moudo> ")
        if command == "help":
            print("Commands:")
            print("  help: Show this help message")
            print("  exit: Exit the toolkit shell")
        elif command == "exit":
            break
        else:
            print(f"Unknown command: {command}")

if __name__ == "__main__":
    app()