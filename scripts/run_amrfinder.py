#!/usr/bin/env python3
"""
BAMPS :: Path C — AMRFinderPlus feature builder
-------------------------------------------------
Runs AMRFinderPlus on a directory of genome FASTA files, collates the
results into a binary (presence/absence) feature matrix suitable for
machine-learning.

Usage (typical):
    python scripts/run_amrfinder.py \
        --genome-dir data/genomes \
        --output-dir outputs/amrfinder \
        --threads 8

Outputs
-------
1. <output-dir>/raw/<sample>.amrfinder.tsv     — raw per-sample results
2. <output-dir>/amr_presence_absence.tsv       — collated 0/1 matrix

The script automatically detects FASTA extensions (.fa, .fas, .fna, .fasta, and their .gz variants) and infers
sample IDs from the base filename (everything before the first dot).  It gracefully skips genomes it has
already processed unless --force is supplied.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import subprocess
from pathlib import Path
from typing import Iterable, List

import pandas as pd

LOG = logging.getLogger("bamps_ml.run_amrfinder")

# Accept common FASTA extensions + their gzip versions (case-insensitive)
FASTA_EXTS = {
    ".fa", ".fas", ".fna", ".fasta",
    ".fa.gz", ".fas.gz", ".fna.gz", ".fasta.gz",
}

###############################################################################
# Helper functions
###############################################################################

def iter_fastas(genome_dir: Path) -> Iterable[Path]:
    """Yield FASTA files in *genome_dir* matching FASTA_EXTS (non‑recursive)."""
    for p in sorted(genome_dir.iterdir()):
        if p.suffix.lower() in FASTA_EXTS or p.name.lower().endswith(tuple(FASTA_EXTS)):
            yield p


def find_genomes(genome_dir: Path) -> List[Path]:
    genomes = list(iter_fastas(genome_dir))
    if not genomes:
        raise FileNotFoundError(f"No FASTA files with extensions {sorted(FASTA_EXTS)} found in {genome_dir}")
    return genomes


def run_amrfinder(genome: Path, out_tsv: Path, threads: int, amrfinder_bin: str):
    """Execute AMRFinderPlus on *genome* (handles gzip transparently)."""
    genome_path = genome
    if genome.suffix == ".gz":
        # AMRFinder can't read gzip; decompress to temp file in same dir
        tmp = genome.with_suffix("")  # strip .gz
        with gzip.open(genome, "rb") as fin, open(tmp, "wb") as fout:
            fout.write(fin.read())
        genome_path = tmp

    cmd = [
        amrfinder_bin,
        "--threads", str(threads),
        "--nucleotide", str(genome_path),
        "--output", str(out_tsv),]
    LOG.debug("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        LOG.error("AMRFinder failed on %s (exit %s)\n%s", genome.name, exc.returncode, exc.stderr.decode())
        raise
    finally:
        if genome_path is not genome and genome_path.exists():
            genome_path.unlink()  # remove temp file

def parse_amrfinder_tsv(tsv: Path) -> pd.Series:
    """Load one AMRFinder TSV and convert to binary feature Series.

    Handles multiple AMRFinder header schemes (e.g., old `#gene_symbol`
    vs newer `Element symbol`).  Any header is lower‑cased and spaces →
    underscores before lookup, so we just search for synonyms.
    """
    df = pd.read_csv(tsv, sep="	", header=0, dtype=str)
    df.columns = [c.lstrip("#").strip().lower().replace(" ", "_") for c in df.columns]

    # Determine which column stores the gene / element symbol
    gene_col = None
    for cand in ("gene_symbol", "element_symbol", "symbol", "name"):
        if cand in df.columns:
            gene_col = cand; break
    if gene_col is None:
        raise KeyError(f"No gene symbol column found in {tsv.name}; header columns: {', '.join(df.columns)}")

    # Identify point mutations vs full genes
    def _feat(r):
        t = str(r.get("type", "")).lower()
        st = str(r.get("subtype", "")).lower()
        if t == "point" or st in {"mutation", "point"}:
            return f"MUT:{r[gene_col]}:{r.get('aa_change', '')}"
        return f"GENE:{r[gene_col]}"

    feats = df.apply(_feat, axis=1).drop_duplicates()
    return pd.Series(1, index=feats)

def build_matrix(tsv_files: List[Path]) -> pd.DataFrame:
    rows = []
    for tsv in tsv_files:
        sample = tsv.stem.split(".")[0]
        series = parse_amrfinder_tsv(tsv)
        series.name = sample
        rows.append(series)
    matrix = pd.DataFrame(rows).fillna(0).astype(int)
    matrix.sort_index(axis=1, inplace=True)
    return matrix

###############################################################################
# Main CLI
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="BAMPS: AMRFinderPlus feature builder")
    parser.add_argument("--genome-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--amrfinder", default="amrfinder")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    genomes = find_genomes(args.genome_dir)
    LOG.info("Found %d genomes", len(genomes))

    produced_tsv = []
    for g in genomes:
        sample = g.stem.split(".")[0]
        out_tsv = raw_dir / f"{sample}.amrfinder.tsv"
        if out_tsv.exists() and not args.force:
            LOG.info("Skipping %s (exists)", sample)
        else:
            LOG.info("Running AMRFinder on %s", sample)
            run_amrfinder(g, out_tsv, args.threads, args.amrfinder)
        produced_tsv.append(out_tsv)

    LOG.info("Collating results → presence/absence matrix")
    matrix = build_matrix(produced_tsv)
    matrix_path = args.output_dir / "amr_presence_absence.tsv"
    matrix.to_csv(matrix_path, sep="\t")
    LOG.info("Saved matrix: %s (shape=%s)", matrix_path, matrix.shape)

if __name__ == "__main__":
    main()