#!/bin/bash

ARGS="--silent-prompt -p "
SYSTEM="You are a generator to create a corpus of text to train a text generation model. You generate content about random diverse topics, formatted as list of sentences with line breaks, following user instructions. You never write anything else in the output than the content of the corpus. Make sure to generate a diverse set of sentences.\n\nGenerate 10 example per user question.\n\nExample: Generate simple english sentences.\nThe cat sat on the mat.\nI like to eat apples.\nShe is reading a book.\nThey went to the park.\nHe plays soccer every day.\nWe are happy to see you.\nThe sun is shining bright.\nBirds fly in the sky.\nMy favorite color is blue.\nThe dog barks loudly."

OUT="llm_generated_corpus.tmp"

# 1. Everyday Life and Common Objects
$MODEL $ARGS "$SYSTEM Write simple English sentences about household items like spoons, plates, cups, and brooms." >> $OUT
$MODEL $ARGS "Describe what you might find in a kitchen using simple English sentences." >> $OUT
$MODEL $ARGS "$SYSTEM Write about common activities people do at home, like cooking, cleaning, or watching TV." >> $OUT
$MODEL $ARGS "$SYSTEM Write simple sentences about things you might see in a park, such as trees, benches, and swings." >> $OUT

# 2. Animals and Nature
$MODEL $ARGS "$SYSTEM Write simple English sentences about farm animals like cows, pigs, and chickens." >> $OUT
$MODEL $ARGS "Describe what animals do in the wild using simple English sentences." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a bird learning to fly for the first time." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a squirrel gathering nuts for the winter." >> $OUT
$MODEL $ARGS "$SYSTEM Write about a day in the life of a butterfly, from caterpillar to flight." >> $OUT

# 3. Adventure and Exploration
$MODEL $ARGS "$SYSTEM Write a story about a dog who gets lost and finds its way back home." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a group of friends exploring a mysterious forest." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a child who discovers a hidden treasure in their backyard." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a spaceship visiting a new planet for the first time." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a fish exploring the ocean and meeting new sea creatures." >> $OUT

# 4. Professions and Jobs
$MODEL $ARGS "$SYSTEM Write simple English sentences about different jobs, like teachers, doctors, and firefighters." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a baker making a special cake for a birthday party." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a farmer working hard to grow crops." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a pilot flying a plane through a storm." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a scientist inventing a new machine." >> $OUT

# 5. Emotions and Relationships
$MODEL $ARGS "$SYSTEM Write simple English sentences about feelings like happiness, sadness, and excitement." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about two friends solving a problem together." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a child overcoming their fear of the dark." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a family going on a fun vacation." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a person helping a stranger and making a new friend." >> $OUT

# 6. Seasons and Weather
$MODEL $ARGS "$SYSTEM Write simple English sentences about the four seasons: spring, summer, autumn, and winter." >> $OUT
$MODEL $ARGS "Describe a sunny day at the beach using simple English sentences." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about building a snowman on a snowy day." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a family enjoying a picnic on a windy day." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a rainy day and how it made everyone stay indoors." >> $OUT

# 7. Food and Cooking
$MODEL $ARGS "$SYSTEM Write simple English sentences about different types of food, like pizza, pasta, and salad." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a child helping their parent bake cookies." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a chef preparing a special meal for a competition." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a family trying a new recipe for dinner." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a farmer growing vegetables and selling them at a market." >> $OUT

# 8. Fantasy and Imagination
$MODEL $ARGS "$SYSTEM Write a story about a dragon who loves to bake cupcakes." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a magical tree that grants wishes." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a unicorn exploring a rainbow-colored forest." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a wizard who accidentally turns their cat into a frog." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a child who finds a talking rock in their garden." >> $OUT

# 9. Travel and Transportation
$MODEL $ARGS "$SYSTEM Write simple English sentences about different types of vehicles, like cars, buses, and trains." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a family going on a road trip." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a train conductor helping passengers on a long journey." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a hot air balloon ride over the mountains." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a child’s first airplane ride." >> $OUT

# 10. Learning and Education
$MODEL $ARGS "$SYSTEM Write simple English sentences about school subjects like math, science, and history." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a student learning to read for the first time." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a teacher helping a shy student gain confidence." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a group of friends working on a school project together." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a child discovering their love for books." >> $OUT

# 11. Holidays and Celebrations
$MODEL $ARGS "$SYSTEM Write simple English sentences about holidays like Christmas, Halloween, and Easter." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a family decorating their house for Christmas." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about children trick-or-treating on Halloween." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a birthday party with games and cake." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a community celebrating a festival with music and food." >> $OUT

# 12. Random Fun Prompts
$MODEL $ARGS "$SYSTEM Write a story about a robot who wants to learn how to dance." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a talking dog who becomes a detective." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a child who finds a map to a hidden island." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a magical pencil that draws real things." >> $OUT
$MODEL $ARGS "$SYSTEM Write a story about a group of animals starting a band." >> $OUT
