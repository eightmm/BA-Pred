#!/bin/bash

# BA-Pred Example Inference Script

if command -v bapred &> /dev/null; then
    bapred \
        -r 1KLT.pdb \
        -l ligands.sdf \
        -o result.tsv \
        --batch_size 64 \
        --device cuda
else
    cd ..
    python -m bapred.inference \
        -r example/1KLT.pdb \
        -l example/ligands.sdf \
        -o example/result.tsv \
        --batch_size 64 \
        --device cuda
    cd example
fi
