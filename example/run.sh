#!/bin/bash
python_path="python"
inference_py="../scripts/run_inference.py"

$python_path $inference_py -r ./1KLT.pdb -l ligands.sdf -o result.tsv
