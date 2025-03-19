# Define the source directory and the destination directory
$sourceDir = Get-Location
$destDir = "$sourceDir/build/toolkit/bin"

# Create the destination directory if it doesn't exist
if (-Not (Test-Path -Path $destDir)) {
    New-Item -ItemType Directory -Path $destDir -Force
}

# Get all .py files except setup.py
$pyFiles = Get-ChildItem -Path $sourceDir -Filter *.py -Recurse | Where-Object { $_.Name -ne 'setup.py' }

# Loop through each .py file and convert it to an executable using pyinstaller
foreach ($file in $pyFiles) {
    pyinstaller --onefile --distpath $destDir $file.FullName
}

Write-Output "All .py files have been converted to executables and placed in $destDir"