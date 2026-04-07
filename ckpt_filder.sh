#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <source_ckpt> <dest_folder>"
    exit 1
fi

SRC="$1"
DST="$2"

REF=$(md5sum "$SRC" | awk '{print $1}')
echo "Source: $SRC ($REF)"
echo "---"

total=$(find "$DST" -type f | wc -l)
echo "Found $total files to compare."
echo "---"

found=0
i=0
while IFS= read -r f; do
    ((i++))
    h=$(md5sum "$f" | awk '{print $1}')
    if [ "$h" = "$REF" ]; then
        echo -e "\rMATCH: $f"
        ((found++))
    else
        printf "\r[%d/%d] Checking: %s" "$i" "$total" "$f"
    fi
done < <(find "$DST" -type f)

echo ""
echo "---"
echo "Done. $found match(es) found out of $total files."