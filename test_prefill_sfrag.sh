#!/bin/bash

OUTPUT_FILE="sfrag_full.log"
> $OUTPUT_FILE  # Clear the file

for warp in $(seq 0 3); do
    for thread in $(seq 0 63); do
        echo "Running thread ${thread}... warp ${warp}"
        ./a.out --thread $thread --warp $warp >> $OUTPUT_FILE 2>&1
    done
done

echo "All threads complete. Output in $OUTPUT_FILE"
