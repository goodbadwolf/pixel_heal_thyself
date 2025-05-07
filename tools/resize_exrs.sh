#!/bin/bash

process_exr() {
    local file="$1"
    local dir=$(dirname "$file")
    local filename=$(basename "$file")
    local temp_file="${dir}/.temp_${filename}"
    
    echo "Processing $file"
    
    # Get original channel names
    local channel_info=$(oiiotool --info:verbose=1 -v "$file" 2>&1 | grep "channel list")
    local channels=$(echo "$channel_info" | sed -E 's/.*channel list: (.*)/\1/')
    
    if [ -n "$channels" ]; then
        channels=$(echo "$channels" | tr -d ' ')
        oiiotool "$file" --resize 50% --chnames "$channels" -o "$temp_file"
    else
        echo "Could not extract channel names, using default preservation"
        oiiotool "$file" --resize 50% --chnames "+" -o "$temp_file"
    fi
    
    if [ $? -eq 0 ]; then
        mv "$temp_file" "$file"
        echo "Successfully resized $file"
    else
        echo "Failed to resize $file"
        rm -f "$temp_file" 2>/dev/null
    fi
}

main() {
    local start_dir="${1:-.}"
    
    echo "Starting to process EXR files in $start_dir"
    
    find "$start_dir" -type f -name "*.exr" -print0 | sort -z | while IFS= read -r -d $'\0' file; do
        process_exr "$file"
    done
    
    echo "All EXR files processed"
}

main "$1"