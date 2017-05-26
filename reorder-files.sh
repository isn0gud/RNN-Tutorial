#! /bin/bash

SRC_DIR=$1
DEST_DIR=$2

if [[ ! -d $SRC_DIR ]]; then
    echo "Source path is not a directory..."
    exit
fi

if [[ ! -d $DEST_DIR ]]; then
    echo "Destination path is not a directory..."
    exit
fi

echo "Moving audio files to new destination..."
for filename in $(find $SRC_DIR -type f -name *.flac); do
    mv $filename $DEST_DIR
done

echo "Processing text files..."
for filename in $(find $SRC_DIR -type f -name "*.txt"); do
    cat $filename | while read line; do
        FILENAME=$(echo $line | cut -f1 -d ' ')
        TRANSCRIPTION=$(echo $line | cut -f2- -d ' ')
        echo -n $TRANSCRIPTION > "$DEST_DIR/$FILENAME.txt"
    done
done
