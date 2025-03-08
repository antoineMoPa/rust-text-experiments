#!/bin/bash

echo "" > corpus.txt

for dir in ./level_*/; do
    # reset level corpus
    > $dir/corpus.corpus
    for file in $dir/*.txt; do
        echo "Processing $file"
        cat $file >> corpus.txt
        cat $file >> $dir/corpus.corpus
    done
done

pushd .
cd level_1/
shuf -r corpus.txt | head -n 120 > corpus.corpus
popd

pushd .
cd level_2/
shuf -r corpus.txt | head -n 360 > corpus.corpus
popd

du -h corpus.txt
