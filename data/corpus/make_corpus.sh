#!/bin/bash

echo "" > corpus.txt

for dir in ./level_*/; do
    # reset level corpus
    echo "" > $dir/corpus.corpus
    for file in $dir/*.txt; do
        echo "Processing $file"
        cat $file >> corpus.txt
        cat $file >> $dir/corpus.corpus
    done
done

pushd .
cd level_3
python3 gen_text.py >> corpus.corpus
popd

du -h corpus.txt
