#!/bin/bash

# Set variables
DOWNLOAD_URL="http://mattmahoney.net/dc/enwik9.zip"
# Get the absolute path of the script directory, regardless of where it's called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
ZIP_FILE="$OUTPUT_DIR/enwik9.zip"
EXTRACTED_FILE="$OUTPUT_DIR/enwik9"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Download the file
echo "Downloading enwik9.zip..."
wget -c "$DOWNLOAD_URL" -O "$ZIP_FILE"

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Error: Download failed"
    exit 1
fi

# Extract the file
echo "Extracting enwik9.zip..."
unzip -o "$ZIP_FILE" -d "$OUTPUT_DIR"

# Check if extraction was successful
if [ $? -ne 0 ]; then
    echo "Error: Extraction failed"
    exit 1
fi

# Clean up zip file
echo "Cleaning up..."
rm "$ZIP_FILE"

echo "Download and extraction complete!"
echo "Data saved to: $EXTRACTED_FILE"