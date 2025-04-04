#!/bin/bash

# Define arrays of subjects, verbs, and objects
subjects=("boy" "dog" "chef" "lion" "squirrel" "bird")
verbs=("kicked" "baked" "chased" "bit" "caught" "ate")
objects=("ball" "cookies" "cat" "man" "dog" "squirrel" "bird" "lion" "fish" "carrot" "pizza" "cake" "cheese" "egg" "apple" "banana")

# Generate combinations
for subject in "${subjects[@]}"; do
  for verb in "${verbs[@]}"; do
    for object in "${objects[@]}"; do

      # Skip weird combos (e.g. chef chased ball)
      if [[ "$verb" == "baked" && "$object" != "cookies" ]]; then continue; fi
      if [[ "$verb" == "kicked" && "$object" != "ball" ]]; then continue; fi
      if [[ "$verb" == "bit" && "$object" != "man" ]]; then continue; fi
      if [[ "$verb" == "chased" && "$object" != "cat" && "$object" != "man" ]]; then continue; fi

      sentence="The $subject $verb the $object."
      question="Who $verb the $object?"
      answer="$subject"

      echo -e "$sentence Q: $question A: the $answer<stop>"
    done
  done
done
