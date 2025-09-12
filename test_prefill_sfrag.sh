#!/bin/bash
# run_all_threads.sh

OUTPUT_FILE="sfrag_combined.log"
> $OUTPUT_FILE  # Clear the file

for thread in {0..63}; do
    echo "Running thread $thread..."
    ./a.out --thread $thread >> $OUTPUT_FILE 2>&1
done

echo "All threads complete. Output in $OUTPUT_FILE"
