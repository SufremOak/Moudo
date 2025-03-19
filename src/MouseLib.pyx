from stdlib cimport system
from stdlib cimport free
from stdlib cimport malloc
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt
from libc.math cimport pow


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

# Check if the system's GPU is Nvidia
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
        sprintf(command, "lspci | grep -i nvidia")
        result = system(command)
        free(command)

    return result