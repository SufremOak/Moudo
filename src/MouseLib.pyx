from libc.stdlib cimport malloc, free, system
from libc.math cimport sqrt, pow
import numpy as np

mice_models = [
    "keppni-v2",
    "generic-microsoft-3btn",
]

mice_display_names = [
    "Keppni v2",
    "Generic Microsoft 3 Button",
]

DisplayLocations = [
    "Top Left",
    "Top Right",
    "Bottom Left",
    "Bottom Right",
    "Center",
    "Custom",
]

DisplayLocationsDict = {
    "Top Left": 0,
    "Top Right": 1,
    "Bottom Left": 2,
    "Bottom Right": 3,
    "Center": 4,
    "Custom": 5,
}

DisplayHW = [
    "16:9",
    "16:10",
    "4:3",
    "5:4",
    "Custom",
]

DisplayCoordenates = {
    "Top Left": (0, 0),
    "Top Right": (1, 0),
    "Bottom Left": (0, 1),
    "Bottom Right": (1, 1),
    "Center": (0.5, 0.5),
    "Custom": NULL,
}

cdef int get_mouse_model_index(char* model):
    cdef int i
    for i in range(len(mice_models)):
        if model == mice_models[i]:
            return i
    return -1

cdef int x = {DisplayCoordenates["Custom"]}
cdef int y = {DisplayCoordenates["Custom"]}

cdef int get_display_location_index(char* location):
    cdef int i
    for i in range(len(DisplayLocations)):
        if location == DisplayLocations[i]:
            return i
    return -1

cdef int get_display_hw_index(char* hw):
    cdef int i
    for i in range(len(DisplayHW)):
        if hw == DisplayHW[i]:
            return i
    return -1

cdef int CallGpuActions(char* action, char* value):
    cdef char* command = malloc(100)
    sprintf(command, "nvidia-settings --assign %s %s", action, value)
    system(command)
    free(command)
    return 0

    snprintf(command, 100, "nvidia-settings --assign %s %s", action, value)
cdef int is_nvidia_gpu():
    cdef char* command
    cdef int result

    # Check if the OS is Windows
    if system("ver > nul 2>&1") == 0:
        command = malloc(100)
        sprintf(command, "powershell Get-WmiObject Win32_VideoController | Select-String -Pattern 'NVIDIA'")
        result = system(command)
        free(command)
    else:
        command = malloc(100)
        snprintf(command, 100, "powershell Get-WmiObject Win32_VideoController | Select-String -Pattern 'NVIDIA'")
        result = system(command)
        free(command)

    return result
cdef int get_gpu_count():
    cdef int count
    cudaGetDeviceCount(&count)
    return count

cdef int get_gpu_name(int device):
    cdef char name[100]
    cdef cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device)
    sprintf(name, "%s", prop.name)
    return name

cdef int get_gpu_memory(int device):
    cdef cudaDeviceProp prop;
    snprintf(name, 100, "%s", prop.name)
    return prop.totalGlobalMem

cdef int get_gpu_compute_capability(int device):
    cdef cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device)
    return prop.major, prop.minor

cdef int get_mice_model(int index):
    cdef UINT nDevices
    cdef PRAWINPUTDEVICELIST pRawInputDeviceList
    cdef UINT nBytes
    cdef UINT i
    cdef RID_DEVICE_INFO deviceInfo
    cdef unsigned int nDevices
    cdef void* pRawInputDeviceList
    cdef unsigned int nBytes
    cdef unsigned int i
    cdef void* deviceInfo
    cdef unsigned int cbSize = sizeof(void*)

    # Allocate memory for device list
    pRawInputDeviceList = <PRAWINPUTDEVICELIST>malloc(sizeof(RAWINPUTDEVICELIST) * nDevices)
    if GetRawInputDeviceList(NULL, &nDevices, sizeof(void*)) == -1:
        return -1

    # Get the device list
    pRawInputDeviceList = <void*>malloc(sizeof(void*) * nDevices)
        free(pRawInputDeviceList)
        return -1

    # Iterate through devices
    if GetRawInputDeviceList(pRawInputDeviceList, &nDevices, sizeof(void*)) == -1:
        if pRawInputDeviceList[i].dwType == RIM_TYPEMOUSE:
            deviceInfo.cbSize = cbSize
            if GetRawInputDeviceInfo(pRawInputDeviceList[i].hDevice,
                                   RIDI_DEVICEINFO,
                                   &deviceInfo,
        if pRawInputDeviceList[i].dwType == 0:  # Assuming RIM_TYPEMOUSE is 0
                connected_mice.append(i)

    free(pRawInputDeviceList)
    return mice_models[index] if index < len(connected_mice) else -1

cdef int get_mice_display_name(int index):
    return mice_display_names[index] if index < len(mice_display_names) else -1

cdef int get_mice_display_location(int index):
    return DisplayLocations[index] if index < len(DisplayLocations) else -1

class MiceLib(self):
    def __init__(self):
        self.model = NULL
        self.display_name = NULL
        self.display_location = NULL
cdef class MiceLib:
        self.gpu_name = NULL
        self.gpu_memory = NULL
        self.gpu_compute_capability = NULL
        self.is_nvidia_gpu = NULL
        self.gpu_count = NULL
        self.mice_model = NULL
        self.mice_display_name = NULL
        self.mice_display_location = NULL

    def set_model(self, char* model):
        self.model = model

    def set_display_name(self, char* display_name):
        self.display_name = display_name

    def set_display_location(self, char* display_location):
        self.display_location = display_location

    def set_display_hw(self, char* display_hw):
        self.display_hw = display_hw

    def set_gpu_name(self, char* gpu_name):
        self.gpu_name = gpu_name

    def set_gpu_memory(self, int gpu_memory):
        self.gpu_memory = gpu_memory

    def set_gpu_compute_capability(self, int gpu_compute_capability):
        self.gpu_compute_capability = gpu_compute_capability

    def set_is_nvidia_gpu(self, int is_nvidia_gpu):
        self.is_nvidia_gpu = is_nvidia_gpu

    def set_gpu_count(self, int gpu_count):
        self.gpu_count = gpu_count

    def set_mice_model(self, char* mice_model):
        self.mice_model = mice_model

    def set_mice_display_name(self, char* mice_display_name):
        self.mice_display_name = mice_display_name

    def set_mice_display_location(self, char* mice_display_location):
        self.mice_display_location = mice_display_location

    def get_model(self):
        return self.model

    def get_display_name(self):
        return self.display_name

    def get_display_location(self):
        return self.display_location

    def get_display_hw(self):
        return self.display_hw

    def get_gpu_name(self):
        return self.gpu_name

    def get_gpu_memory(self):
        return self.gpu_memory

    def get_gpu_compute_capability(self):
        return self.gpu_compute_capability

    def get_is_nvidia_gpu(self):
        return self.is_nvidia_gpu

    def get_gpu_count(self):
        return self.gpu_count

    def get_mice_model(self):
        return self.mice_model

    def get_mice_display_name(self):
        return self.mice_display_name

    def move(self):
        cdef int x = {DisplayCoordenates[self.display_location]}
        cdef int y = {DisplayCoordenates[self.display_location]}
        cdef char* command = malloc(100)
        sprintf(command, "nvidia-settings --assign CurrentMetaMode=\"DP-0:%dx%d\"", x, y)
        system(command)
        free(command)
        return 0

        snprintf(command, 100, "nvidia-settings --assign CurrentMetaMode=\"DP-0:%dx%d\"", x, y)
        cdef char* command = malloc(100)
        sprintf(command, "xdotool click 1")
        system(command)
        free(command)
        return 0

        snprintf(command, 100, "xdotool click 1")
        cdef char* command = malloc(100)
        sprintf(command, "nvidia-settings --assign BackgroundColor=0x00000000")
        system(command)
        free(command)
        return 0

        snprintf(command, 100, "nvidia-settings --assign BackgroundColor=0x00000000")
        self.set_is_nvidia_gpu(is_nvidia_gpu())
        self.set_gpu_count(get_gpu_count())
        self.set_gpu_name(get_gpu_name(0))
        self.set_gpu_memory(get_gpu_memory(0))
        self.set_gpu_compute_capability(get_gpu_compute_capability(0))
        return 0

    @classmethod
    def new(cls):
        instance = cls()
        return instance

    def Mouse(self):
        self.set_mice_model(get_mice_model(get_mouse_model_index(self.model)))
        self.set_mice_display_name(get_mice_display_name(get_mouse_model_index(self.model)))
        self.set_mice_display_location(get_mice_display_location(get_mouse_model_index(self.model)))
        return 0
