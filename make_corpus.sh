#!/bin/bash

cd data/corpus/

for dir in ./level_*/; do
    # reset level corpus
    > $dir/corpus.corpus
    for file in $dir/*.txt; do
        echo "Processing $file"
        cat $file >> $dir/corpus.corpus
    done
done

pushd .
cd level_1/
shuf -r corpus.txt | head -n 120 > corpus.corpus
popd

pushd .
cd level_2/
shuf -r corpus.txt | head -n 500 > corpus.corpus
popd

pushd .
cd level_3/
shuf ../level_2/corpus.txt > corpus.1.corpus
shuf ../level_2/corpus.txt >> corpus.1.corpus
shuf ../level_2/corpus.txt >> corpus.1.corpus
shuf corpus.txt >> corpus.1.corpus
shuf corpus.1.corpus > corpus.corpus
shuf corpus.1.corpus >> corpus.corpus
shuf corpus.1.corpus >> corpus.corpus

du -h corpus.corpus
popd
