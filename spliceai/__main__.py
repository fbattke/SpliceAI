import sys
import argparse
import logging
import pysam
from spliceai.utils import Annotator, get_delta_scores
from tqdm import tqdm
from pathlib import Path
from time import time


try:
    from sys.stdin import buffer as std_in
    from sys.stdout import buffer as std_out
except ImportError:
    from sys import stdin as std_in
    from sys import stdout as std_out


def get_options():

    parser = argparse.ArgumentParser(description='Version: 1.3.1')
    parser.add_argument('-I', metavar='input', nargs='?', default=std_in,
                        help='path to the input VCF file, defaults to standard in')
    parser.add_argument('-O', metavar='output', nargs='?', default=std_out,
                        help='path to the output VCF file, defaults to standard out')
    parser.add_argument('-R', metavar='reference', required=True,
                        help='path to the reference genome fasta file')
    parser.add_argument('-A', metavar='annotation', required=True,
                        help='"grch37" (GENCODE V24lift37 canonical annotation file in '
                             'package), "grch38" (GENCODE V24 canonical annotation file in '
                             'package), or path to a similar custom gene annotation file')
    parser.add_argument('-D', metavar='distance', nargs='?', default=50,
                        type=int, choices=range(0, 5000),
                        help='maximum distance between the variant and gained/lost splice '
                             'site, defaults to 50')
    parser.add_argument('-M', metavar='mask', nargs='?', default=0,
                        type=int, choices=[0, 1],
                        help='mask scores representing annotated acceptor/donor gain and '
                             'unannotated acceptor/donor loss, defaults to 0')
    parser.add_argument('-S', metavar='skip', dest="skip_chr", default=None,
                        help='Skip variants with the assigned chromosomes. '
                             'The input will be splitted into chromosome numbers based on the commas in the input string.'
                             'Ignored if this argument is not used.'
                             'Example: "1", "1,2,3,M", and "M,Y"')
    args = parser.parse_args()

    return args


def format_vcf_record(record) -> str:
    return f"{record.chrom}_{record.pos}_{record.ref}_{record.alts}"


def read_outputs(vcf_file):
    records = []
    for record in vcf_file:
        records.append(format_vcf_record(record))
    return records


def main():

    args = get_options()

    if None in [args.I, args.O, args.D, args.M]:
        logging.error('Usage: spliceai [-h] [-I [input]] [-O [output]] -R reference -A annotation '
                      '[-D [distance]] [-M [mask]]')
        exit()

    try:
        vcf = pysam.VariantFile(args.I)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()

    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')
    skip_chrs = []
    if args.skip_chr is not None:
        skip_chrs = args.skip_chr.split(",")
        print(f"Variants located on the chromosome {skip_chrs} will not be scored.")

    precomputed_outputs = []
    prev_output = None
    if Path(args.O).is_file():
        prev_output = [record.copy() for record in pysam.VariantFile(args.O, header=header)]
        precomputed_outputs = read_outputs(prev_output)
        print(f"Found precomputed {len(precomputed_outputs)} records")
    try:
        output = pysam.VariantFile(args.O, mode='w', header=header)
    except (IOError, ValueError) as e:
        logging.error('{}'.format(e))
        exit()
    ann = Annotator(args.R, args.A)

    if len(precomputed_outputs) > 0:
        for record in prev_output:
            output.write(record)
    n_total_vcfs = len(vcf)

    start_t = time()
    n_actual, n_skip_chr, n_skip_seq, n_skip_precomputed = 0, 0, 0, 0
    for record in tqdm(vcf, total=n_total_vcfs, desc="Number of variants"):
        print(format_vcf_record(record))
        if format_vcf_record(record) in precomputed_outputs:
            n_skip_precomputed += 1
            continue

        print(f"{record.chrom}")
        if f"{record.chrom}" in skip_chrs:
            output.write(record)
            n_skip_chr += 1
            continue
        print(f"{record.chrom}")
        scores = get_delta_scores(record, ann, args.D, args.M)
        if len(scores) > 0:
            record.info['SpliceAI'] = scores
            n_actual += 1
        else:
            n_skip_seq += 1
        output.write(record)
    end_t = time()

    print(f"Finished the whole process in {end_t - start_t} secs.")
    print(f"Skipped {n_skip_precomputed} precomputed results.")
    print(f"Skipped {n_skip_chr} variants on the skipped chromosomes.")
    print(f"Skipped {n_skip_seq} variants dues to sequence or reference genome issues.")
    print(f"Scored {n_actual} variants")
    if n_actual != 0:
        print(f"Elapsed time per variant (actually calculated): {(end_t - start_t) / n_actual}")
    
    vcf.close()
    output.close()


if __name__ == '__main__':
    main()
