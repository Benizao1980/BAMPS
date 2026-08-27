# Input formats

## Feature matrix

Delimited text (TSV recommended), one row per sample. The first identifier column should contain stable isolate/sample IDs and all remaining model features should be numeric or binary.

## Phenotype table

Delimited text with the same sample IDs and one column per antibiotic. Quantitative MICs must be positive numeric values. Missing labels are permitted and are dropped per antibiotic.

## External validation

External validation data must be transformed using the same feature definitions as the training data. Do not select GWAS features using the external validation outcomes.
