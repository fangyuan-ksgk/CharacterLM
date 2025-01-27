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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
ZIP_FILE="$OUTPUT_DIR/enwik9.zip"
EXTRACTED_FILE="$OUTPUT_DIR/enwik9"

# Create Perl filter script
cat > "$OUTPUT_DIR/wikifil.pl" << 'EOF'
#!/usr/bin/perl
# Program to filter Wikipedia XML dumps to clean text
$/=">";
while (<>) {
  if (/<text /) {$text=1;}
  if (/#redirect/i) {$text=0;}
  if ($text) {
    if (/<\/text>/) {$text=0;}
    s/<.*>//;
    s/&amp;/&/g;
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;
    s/<[^>]*>//g;
    s/\[http:[^] ]*/[/g;
    s/\|thumb//ig;
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;
    s/\[\[[^\|\]]*\|/[[/g;
    s/{{[^}]*}}//g;
    s/{[^}]*}//g;
    s/\[//g;
    s/\]//g;
    s/&[^;]*;/ /g;
    $_=" $_ ";
    tr/A-Z/a-z/;
    s/0/ zero /g;
    s/1/ one /g;
    s/2/ two /g;
    s/3/ three /g;
    s/4/ four /g;
    s/5/ five /g;
    s/6/ six /g;
    s/7/ seven /g;
    s/8/ eight /g;
    s/9/ nine /g;
    tr/a-z/ /cs;
    chop;
    print $_;
  }
}
EOF

# Make the Perl script executable
chmod +x "$OUTPUT_DIR/wikifil.pl"

# Run the filter and save the output
echo "Filtering text..."
perl "$OUTPUT_DIR/wikifil.pl" "$EXTRACTED_FILE" > "$OUTPUT_DIR/filtered_enwiki9.txt"

echo "Text filtering complete!"
echo "Filtered text saved to: $OUTPUT_DIR/filtered_enwiki9.txt"