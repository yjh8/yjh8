# Script to recursively copy all ".py" files from Folder A to Folder B

# Define source and destination folders
$sourceFolder = "C:\Path\To\FolderA"
$destinationFolder = "C:\Path\To\FolderB"

# Ensure the destination folder exists
if (!(Test-Path -Path $destinationFolder)) {
    New-Item -ItemType Directory -Path $destinationFolder
}

# Get all .py files from the source folder and its subfolders
$files = Get-ChildItem -Path $sourceFolder -Recurse -Filter "*.py"

# Copy each file to the destination folder, preserving the directory structure
foreach ($file in $files) {
    $relativePath = $file.FullName.Substring($sourceFolder.Length).TrimStart("\")
    $destinationPath = Join-Path -Path $destinationFolder -ChildPath $relativePath

    # Ensure the destination subfolder exists
    $destinationSubfolder = Split-Path -Path $destinationPath -Parent
    if (!(Test-Path -Path $destinationSubfolder)) {
        New-Item -ItemType Directory -Path $destinationSubfolder
    }

    # Copy the file
    Copy-Item -Path $file.FullName -Destination $destinationPath -Force
}