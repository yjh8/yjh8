#!/bin/bash
# Script to analyze a directory and calculate the ratio of uncompressible file types by size.

# --- Configuration ---
# List of extensions considered "Pre-Compressed" or "Uncompressible" (Case-insensitive matching)
# These files are typically already highly compressed and yield minimal further compression gains.
UNCOMPRESSIBLE_EXTS="jpg jpeg png gif webp heic mp4 mov avi mkv wmv mp3 flac m4a ogg aac zip rar 7z gz xz bz2 zst iso dmg"

# --- Variables ---
TOTAL_SIZE=0
UNCOMPRESSIBLE_SIZE=0
COMPRESSIBLE_SIZE=0

# --- Function to display usage information ---
show_usage() {
    echo "Usage: $0 [DIRECTORY_PATH]"
    echo ""
    echo "Calculates the size ratio of uncompressible (video, audio, compressed files) to"
    echo "compressible file types in a given directory (recursively)."
    echo ""
    echo "Arguments:"
    echo "  DIRECTORY_PATH  The path to the directory to analyze. Defaults to the current directory (.)."
    echo "  -h, --help      Display this help message."
    echo ""
}

# --- Helper function for human-readable output ---
# Requires 'numfmt' or equivalent, but we'll stick to printing bytes for calculation consistency.
bytes_to_human() {
    local bytes=$1
    if command -v numfmt &> /dev/null; then
        numfmt --to=iec --suffix=B "$bytes"
    else
        echo "$bytes bytes"
    fi
}

# --- Main Logic ---

# Check for help flags or too many arguments
if [ "$#" -gt 1 ] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
    show_usage
    exit 0
fi

# Source directory to analyze (Defaults to current directory if no argument is passed)
TARGET_DIR="${1:-.}"

echo "Analyzing file types in: '$TARGET_DIR'"
echo "--------------------------------------------------------"

# Check if TARGET_DIR exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist."
    show_usage
    exit 1
fi

# Find all files recursively and process them
# -print0 and xargs -0 are used for safe handling of filenames with spaces or special characters.
find "$TARGET_DIR" -type f -print0 | while IFS= read -r -d $'\0' file; do
    # Get file size in bytes (using GNU/BSD stat which is reliable on macOS)
    # The output format for size is '%z' on BSD/macOS. For Linux it is usually '-c%s'.
    # We will use 'stat -f %z' as it is generally reliable on macOS.
    file_size=$(stat -f %z "$file" 2>/dev/null)
    
    # Skip if size is zero or stat failed
    if [ -z "$file_size" ] || [ "$file_size" -eq 0 ]; then
        continue
    fi
    
    # Add to total size
    TOTAL_SIZE=$((TOTAL_SIZE + file_size))
    
    # Extract lowercase extension
    filename=$(basename "$file")
    ext="${filename##*.}"
    ext_lc=$(echo "$ext" | tr '[:upper:]' '[:lower:]')

    # Check if extension is in the uncompressible list (using case-insensitive regex match)
    if [[ " $UNCOMPRESSIBLE_EXTS " =~ " $ext_lc " ]]; then
        UNCOMPRESSIBLE_SIZE=$((UNCOMPRESSIBLE_SIZE + file_size))
    else
        COMPRESSIBLE_SIZE=$((COMPRESSIBLE_SIZE + file_size))
    fi

done

# --- Results Calculation and Display ---

if [ "$TOTAL_SIZE" -eq 0 ]; then
    echo "No files found for analysis."
    exit 0
fi

# Calculate percentage
UNCOMPRESSIBLE_PERCENT=$(awk "BEGIN {printf \"%.2f\", ($UNCOMPRESSIBLE_SIZE / $TOTAL_SIZE) * 100}")
COMPRESSIBLE_PERCENT=$(awk "BEGIN {printf \"%.2f\", ($COMPRESSIBLE_SIZE / $TOTAL_SIZE) * 100}")

echo "Analysis Complete:"
echo "--------------------------------------------------------"
echo "TOTAL FILESYSTEM SIZE:    $(bytes_to_human $TOTAL_SIZE)"
echo "--------------------------------------------------------"
echo "UNCOMPRESSIBLE DATA (e.g., Photos, Videos, Archives):"
echo "  Size:      $(bytes_to_human $UNCOMPRESSIBLE_SIZE)"
echo "  Portion:   $UNCOMPRESSIBLE_PERCENT%"
echo "--------------------------------------------------------"
echo "COMPRESSIBLE DATA (e.g., Text, Code, Logs, Documents):"
echo "  Size:      $(bytes_to_human $COMPRESSIBLE_SIZE)"
echo "  Portion:   $COMPRESSIBLE_PERCENT%"
echo "--------------------------------------------------------"

if (( $(echo "$UNCOMPRESSIBLE_PERCENT > 50" | bc -l) )); then
    echo "💡 Note: Over 50% of your data is likely pre-compressed. Expect moderate overall compression gains."
else
    echo "🚀 Note: The majority of your data is compressible. Expect high compression efficiency!"
fi