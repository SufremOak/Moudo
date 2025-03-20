# MouseLib.pyx

from libc.stdint cimport int32_t
import ctypes

# Load the DLL that interacts with the mouse hardware
mouse_dll = ctypes.CDLL("path_to_mouse_dll.dll")

# Define constants for mouse buttons
LEFT_BUTTON = 1
RIGHT_BUTTON = 2
MIDDLE_BUTTON = 4
BUTTON_4 = 8
BUTTON_5 = 16

cdef extern from "windows.h":
    cdef int CreateWindowExA(unsigned long, const char*, const char*, unsigned long, int, int, int, int, void*, void*, void*, void*)
    cdef int ShowWindow(void*, int)
    cdef int UpdateWindow(void*)
    cdef int DefWindowProcA(void*, unsigned int, unsigned long, long)
    cdef int RegisterClassExA(void*)
    cdef int GetModuleHandleA(const char*)
    cdef int LoadCursorA(void*, const char*)
    cdef int LoadIconA(void*, const char*)
    cdef int PostQuitMessage(int)
    cdef int GetMessageA(void*, void*, unsigned int, unsigned int)
    cdef int TranslateMessage(void*)
    cdef int DispatchMessageA(void*)
    cdef int WNDCLASSEXA

cdef class Mouse:
    cdef int x, y
    cdef int button_state

    def __init__(self):
        self.x = 0
        self.y = 0
        self.button_state = 0

    def move(self, int dx, int dy):
        """
        Move the mouse cursor by (dx, dy).
        """
        self.x += dx
        self.y += dy
        mouse_dll.move_mouse(dx, dy)

    def click(self, int button):
        """
        Click a mouse button.
        """
        self.button_state |= button
        mouse_dll.click_button(button)

    def release(self, int button):
        """
        Release a mouse button.
        """
        self.button_state &= ~button
        mouse_dll.release_button(button)

    def scroll(self, int delta):
        """
        Scroll the mouse wheel.
        """
        mouse_dll.scroll_wheel(delta)

    def move_to(self, int x, int y):
        """
        Move the mouse cursor to an absolute position (x, y).
        """
        self.x = x
        self.y = y
        mouse_dll.move_mouse_to(x, y)

    def double_click(self, int button):
        """
        Perform a double click with a mouse button.
        """
        self.click(button)
        self.release(button)
        self.click(button)
        self.release(button)

    def get_position(self):
        """
        Get the current mouse cursor position.
        """
        return self.x, self.y

    def get_button_state(self):
        """
        Get the current state of the mouse buttons.
        """
        return self.button_state

    def is_button_pressed(self, int button):
        """
        Check if a specific mouse button is pressed.
        """
        return (self.button_state & button) != 0

cdef class Window:
    cdef void* hwnd

    def __init__(self, const char* title, int width, int height):
        self.hwnd = self.create_window(title, width, height)

    cdef void* create_window(self, const char* title, int width, int height):
        wc = WNDCLASSEXA()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXA)
        wc.style = 0
        wc.lpfnWndProc = DefWindowProcA
        wc.cbClsExtra = 0
        wc.cbWndExtra = 0
        wc.hInstance = GetModuleHandleA(None)
        wc.hIcon = LoadIconA(None, 32512)  # IDI_APPLICATION
        wc.hCursor = LoadCursorA(None, 32512)  # IDC_ARROW
        wc.hbrBackground = 5  # COLOR_WINDOW + 1
        wc.lpszMenuName = None
        wc.lpszClassName = b"WindowClass"
        wc.hIconSm = LoadIconA(None, 32512)  # IDI_APPLICATION

        if not RegisterClassExA(ctypes.byref(wc)):
            raise WindowsError("Failed to register window class")

        hwnd = CreateWindowExA(
            0,
            wc.lpszClassName,
            title,
            0xCF0000,  # WS_OVERLAPPEDWINDOW
            0x80000000,  # CW_USEDEFAULT
            0x80000000,  # CW_USEDEFAULT
            width,
            height,
            None,
            None,
            wc.hInstance,
            None
        )

        if not hwnd:
            raise WindowsError("Failed to create window")

        ShowWindow(hwnd, 1)  # SW_SHOWNORMAL
        UpdateWindow(hwnd)

        return hwnd

    def show(self):
        ShowWindow(self.hwnd, 1)  # SW_SHOWNORMAL
        UpdateWindow(self.hwnd)

    def close(self):
        PostQuitMessage(0)

    def process_messages(self):
        msg = ctypes.create_string_buffer(48)  # MSG structure size
        while GetMessageA(ctypes.byref(msg), None, 0, 0):
            TranslateMessage(ctypes.byref(msg))
            DispatchMessageA(ctypes.byref(msg))