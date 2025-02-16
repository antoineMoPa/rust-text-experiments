#!/bin/bash

echo "" > corpus.txt

for dir in ./level_*/; do
    # reset level corpus
    echo "" >> $dir/corpus.txt
    for file in $dir/*.txt; do
        # if file is corpus.txt, skip
        if [ $file == $dir/corpus.txt ]; then
            continue
        fi

        echo "Processing $file"
        cat $file >> corpus.txt
        cat $file >> $dir/corpus.txt
    done
done

pushd .
cd level_3
python3 gen_text.py >> corpus.txt
popd

du -h corpus.txt
