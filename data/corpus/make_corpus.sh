#!/bin/bash

cat ./generated_curated_corpus.txt > corpus.txt

for file in ./text_dumps/*; do
    echo "Processing $file"
    cat $file >> corpus.txt
done

python3 gen_text.py >> corpus.txt

# cat wiki-horse.txt >> corpus.txt

du -h corpus.txt
