#!/bin/bash
for file in input_images/*
do
    filename=$(basename -- "$file")
    basename="${filename%.*}"
    python animate.py "$file" 300 "output/$basename.mp4" "output/$basename.txt" -r
    python animate.py "$file" 300 "output/${basename}q.mp4" "output/$basename.txt" -r -q
    echo "$filename"
    echo "$basename"
done
