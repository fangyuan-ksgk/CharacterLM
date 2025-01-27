SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
ZIP_FILE="$OUTPUT_DIR/enwik9.zip"
EXTRACTED_FILE="$OUTPUT_DIR/enwik9"

# Check if input file exists
if [ ! -f "$EXTRACTED_FILE" ]; then
    echo "Error: Input file $EXTRACTED_FILE not found!"
    exit 1
fi

# Create Perl filter script
cat > "$OUTPUT_DIR/wikifil.pl" << 'EOF'
#!/usr/bin/perl
# Program to filter Wikipedia XML dumps to clean text
use strict;
use warnings;

my $text = 0;
$/=">";                     # input record separator
while (<>) {
  if (/<text /) {$text=1;}  # remove all but between <text> ... </text>
  if (/#redirect/i) {$text=0;}  # remove #REDIRECT
  if ($text) {

    # Remove any text not normally visible
    if (/<\/text>/) {$text=0;}
    s/<.*>//;               # remove xml tags
    s/&amp;/&/g;            # decode URL encoded chars
    s/&lt;/</g;
    s/&gt;/>/g;
    s/<ref[^<]*<\/ref>//g;  # remove references <ref...> ... </ref>
    s/<[^>]*>//g;           # remove xhtml tags
    s/\[http:[^] ]*/[/g;    # remove normal url, preserve visible text
    s/\|thumb//ig;          # remove images links, preserve caption
    s/\|left//ig;
    s/\|right//ig;
    s/\|\d+px//ig;
    s/\[\[image:[^\[\]]*\|//ig;
    s/\[\[category:([^|\]]*)[^]]*\]\]/[[$1]]/ig;  # show categories without markup
    s/\[\[[a-z\-]*:[^\]]*\]\]//g;  # remove links to other languages
    s/\[\[[^\|\]]*\|/[[/g;  # remove wiki url, preserve visible text
    s/{{[^}]*}}//g;         # remove {{icons}} and {tables}
    s/{[^}]*}//g;
    s/\[//g;                # remove [ and ]
    s/\]//g;
    s/&[^;]*;/ /g;          # remove URL encoded chars

    # Keep only specified characters
    tr/!$&',-.:;?A-Za-z//cd;    # Delete all characters except the ones specified
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