"""Command-line interface for BA-Pred."""

from pathlib import Path
import argparse
import sys

from bapred.weights import DEFAULT_MODEL, MODEL_PRESETS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BA-Pred: Protein-ligand Binding Affinity Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  bapred -r protein.pdb -l ligands.sdf -o results.tsv\n"
            "  bapred -r protein.pdb -l ligands.sdf -o results.tsv --device cpu\n"
            "  bapred -r protein.pdb -l ligands.sdf -o results.tsv --model random_seed1\n"
            "  python scripts/run_inference.py -r example/1KLT.pdb "
            "-l example/ligands.sdf -o results.tsv"
        ),
    )
    parser.add_argument("-r", "--protein_pdb", required=True, help="Receptor protein PDB file path")
    parser.add_argument("-l", "--ligand_file", required=True, help="Ligand file path (.sdf/.mol2/.txt)")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file path for results")
    parser.add_argument("--batch_size", default=16, type=int, help="Batch size for inference (default: 16)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to use: cpu or cuda (default: cuda)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        choices=sorted(MODEL_PRESETS),
        help=f"Packaged model preset to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Optional checkpoint file or directory. Overrides --model when set.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    from bapred.inference import inference
    from bapred.logger import setup_logger
    import torch

    logger = setup_logger("bapred.cli")

    if not Path(args.protein_pdb).exists():
        logger.error("Protein PDB file not found: %s", args.protein_pdb)
        raise SystemExit(1)

    if not Path(args.ligand_file).exists():
        logger.error("Ligand file not found: %s", args.ligand_file)
        raise SystemExit(1)

    if args.model_path is not None:
        weight_file = Path(args.model_path)
        if weight_file.is_dir():
            weight_file = weight_file / "BAPred.pth"
        if not weight_file.exists():
            logger.error("Model weights not found: %s", weight_file)
            raise SystemExit(1)

    if args.device == "cuda" and torch.cuda.is_available():
        device = "cuda"
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    elif args.device == "cuda":
        device = "cpu"
        logger.warning("CUDA not available, falling back to CPU")
    else:
        device = "cpu"
        logger.info("Using CPU")

    logger.info("Input protein: %s", args.protein_pdb)
    logger.info("Input ligands: %s", args.ligand_file)
    logger.info("Output file: %s", args.output)
    logger.info("Batch size: %s", args.batch_size)
    logger.info("Device: %s", device)
    logger.info("Model preset: %s", args.model)
    logger.info("-" * 50)

    try:
        inference(
            protein_pdb=args.protein_pdb,
            ligand_file=args.ligand_file,
            output=args.output,
            batch_size=args.batch_size,
            model=args.model,
            model_path=args.model_path,
            device=device,
        )
    except Exception as exc:
        logger.error("Error during inference: %s", exc)
        raise SystemExit(1) from exc

    logger.info("Inference completed successfully")
    logger.info("Results saved to: %s", args.output)


if __name__ == "__main__":
    main(sys.argv[1:])
