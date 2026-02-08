#!/bin/bash

# mpf → JPEG 自動抽出スクリプト
# 使い方: ./mpf_extract.sh *.mpf

if [ $# -eq 0 ]; then
    echo "Usage: $0 <mpf files>"
    exit 1
fi

for file in "$@"; do
    echo "Processing: $file"

    # MPImage の数を確認
    count=$(exiftool "$file" | grep -c "MP Image [1-9][0-9]*")

    if [ "$count" -eq 0 ]; then
        echo "  No MPImage found in $file"
        continue
    fi

    echo "  Found $count MPImage entries"

    for tag in $(exiftool "$file" | grep 'MP Image' | awk -F: '{print $1}' | sed -e 's/ //g' | grep '[0-9]$'); do
        output="${file%.*}_${tag}.jpg"
        echo "  Extracting $tag → $output"
        exiftool -b -$tag "$file" > "$output"
    done

    echo "  Done."
    echo
done
