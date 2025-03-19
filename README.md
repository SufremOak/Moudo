# Moudo - Python-based Library for Programming Computer Mice

**Moudo** is a powerful and flexible library designed for direct programming of computer mice. Whether you want to automate repetitive tasks, create custom macros, or push the limits of your mouse’s capabilities, Moudo provides an intuitive interface for controlling your mouse.

## Features

- **Hybrid Approach**:

  - Write your scripts in **`.moudo`** (scripting format).
  - Compile your scripts into **`.mouc`** (compiled format) for improved performance.

- **NVIDIA GPU Support**:

  - If your system has an NVIDIA GPU, Moudo leverages **CUDA** to provide optimized controls and functionality for faster execution.

- **Cross-platform**:

  - Initially for **Windows**, with plans to extend to **Linux** and **macOS**.

- **First Mouse Supported**:

  - **Evolut EG-111 KEPPNI-V2** mouse.
  - **Generic 3btn** mouse (partial)

## Installation

To install **Moudo**, you can use the following command:

```bash
powershell -c "irm sufremoak.github.io/moudo/scripts/install.ps1 | iex"
```

**For systems with an NVIDIA GPU**, make sure you have **CUDA** installed for enhanced performance.

## Quick Start

Here’s how to create a basic script with **Moudo**:

1. Write a **.moudo** script to define your mouse actions.

   Example (`my_script.moudo`):

   ```python
   from moudo import Moudo

   # Initialize the mouse
   mouse = Moudo()

   # Move the mouse to coordinates (500, 500)
   mouse.move_to(500, 500)

   # Left-click
   mouse.click_left()

   # Right-click
   mouse.click_right()
   ```

2. Compile the script to **.mouc** (optional):

   ```bash
   moudo compile my_script.moudo
   ```

3. Run the compiled file:

   ```bash
   moudo run my_script.mouc
   ```

## Special GPU Behavior

If you have an **NVIDIA GPU** on your system, Moudo will automatically detect and use **CUDA** for optimized mouse control. This provides a performance boost for more complex operations. Otherwise, the library will default to the regular CPU-based controls.

## Supported Mice

- **Evolut EG-111 KEPPNI-V2**

This is the first mouse supported by Moudo, and it will receive full functionality through the library, including high precision movement and DPI adjustments.

## Future Features

- Additional mouse models support.
- Expanded CUDA capabilities for other input devices.
- Multi-platform support for **Linux** and **macOS**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Feel free to contribute! Open issues and pull requests are always welcome. If you have suggestions or requests for additional mouse support or features, please open an issue.
