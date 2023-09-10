# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NoneType is deprecated
# from types import NoneType
NoneType = type(None)

from typing import Union
import os
import argparse
from functools import partial
from tqdm import tqdm

import wandb
import torch
import numpy as np
import sklearn.metrics as metrics

from datasets import Dataset, Sequence
from transformers import DataCollatorWithPadding

from utils.submitit import str2bool  # better bool flag type for argparse
from utils.io import read_jsonlines, read_json, write_json, write_jsonlines
from utils.notebooks import filter_text_col_length, infer_length_column

from utils.evaluation import (
    SUPPORTED_METRICS,
    NO_CHECK_ARGS,
    ROC_TEST_STAT_SUFFIXES,
    FILTER_BY_COLUMNS,
    conditional_no_check_args,
    load_oracle_model,
    evaluate_ppl,
    load_detector,
    compute_z_scores,
    compute_windowed_z_scores,
    compute_run_len_chsqrd_stats,
    compute_repetition_diversity,
    compute_p_sp,
    compute_coherence,
    compute_mauve,
    compute_detect_retrieval,
    load_tokenizer,
    concat_rows,
)

print(f"Current huggingface cache dir: {os.environ['HF_HOME']}")

from datasets import disable_caching

disable_caching()


def main(args):



    for run_name, input_dir, output_dir in zip(args.run_name, args.input_dir, args.output_dir):
        ###########################################################################
        # Create output dir if it doesn't exist, and warn if it contains metric file
        ###########################################################################
        gen_table_w_metrics_path = f"{output_dir}/gen_table_w_metrics.jsonl"
        metrics_meta_path = f"{output_dir}/gen_table_w_metrics_meta.json"

        print(f"Output dir for this run: {output_dir}")
        # notify if exists
        if os.path.exists(output_dir):
            print(f"Output dir for this run already exists!")
            print(f"Contents: {sorted(os.listdir(output_dir))}")
            # warn if metrics file exists
            if os.path.exists(gen_table_w_metrics_path):
                if not args.overwrite_output_file:
                    print(
                        f"WARNING: Exiting to avoid overwriting output file. "
                        f"Pass the '--overwrite_output_file' flag to ignore this check."
                    )
                    exit()
                else:
                    print(
                        f"WARNING: Found existing generation files with metrics added at this output dir. "
                        f"Overwriting anyway :/"
                    )
        else:
            # create the output dir where run artifacts are stored
            os.makedirs(output_dir)

        ###########################################################################
        # Parse metrics to log - ppl, zscore, etc
        ###########################################################################
        print(f"Evaluation metrics to compute: {args.evaluation_metrics}")

        ###########################################################################
        # Load generations
        ###########################################################################
        print(f"Input dir for this run: {input_dir}")
        print(f"Loading previously generated outputs for evaluation via oracle model and metrics...")

        # check for the "attacked version" of the gen table first
        gen_table_meta_path = f"{input_dir}/gen_table_attacked_meta.json"
        gen_table_path = f"{input_dir}/gen_table_attacked.jsonl"
        safe_gen_table_path = f"{input_dir}/gen_table_attacked_safe.jsonl"
        loaded_attacked = True

        attack_variants_exist = [
            os.path.exists(gen_table_meta_path),
            os.path.exists(gen_table_path),
        ]
        if not all(attack_variants_exist):
            loaded_attacked = False
            gen_table_meta_path = f"{input_dir}/gen_table_meta.json"
            gen_table_path = f"{input_dir}/gen_table.jsonl"
            safe_gen_table_path = f"{input_dir}/gen_table_safe.jsonl"

            assert os.path.exists(
                gen_table_meta_path
            ), f"failed file check for prev generations metadata json file: {gen_table_meta_path}"
            assert os.path.exists(
                gen_table_path
            ), f"failed file check for prev generations jsonl file: {gen_table_path}"

        assert not os.path.exists(safe_gen_table_path), (
            f"failed for safety bc there is a secondary 'safe' marked file",
            f" in this dir indicating a possible issue with the generation step. ",
        )

        cmdline_args = args.__dict__.copy()
        prev_gen_table_meta = read_json(gen_table_meta_path)
        joined_args = prev_gen_table_meta.copy()
        for k, v in cmdline_args.items():
            if v is not None:
                joined_args.update({k: v})
            else:
                print(
                    f"cmdline arg {k} is None, leaving it as the value found in the input metadata: {prev_gen_table_meta[k]}"
                )

        # check that the args used to generate the prev generations are the same as
        # the current args, for the intersection of keys
        if not args.overwrite_args:
            # update the no check args based on the current state of args
            current_no_check_args = conditional_no_check_args(
                NO_CHECK_ARGS, args.evaluation_metrics, args
            )

            for key in prev_gen_table_meta.keys():
                if key in current_no_check_args:
                    continue
                assert joined_args[key] == prev_gen_table_meta[key], (
                    f"failed for safety bc after merging the prev metadata with "
                    f"the current cmdline args, values for '{key}' are not the same. "
                    f"in metadata: {prev_gen_table_meta[key]}, passed: {cmdline_args[key]}. "
                    f"Pass the '--overwrite_args' flag to ignore this check."
                )

        args = argparse.Namespace(**joined_args)
        gen_table = [ex for ex in read_jsonlines(gen_table_path)]
        if args.debug:
            gen_table = gen_table[:50]
        if args.limit_rows == -1:
            gen_table_ds = Dataset.from_list(gen_table)
        else:
            gen_table_ds = Dataset.from_list(gen_table[: args.limit_rows])

        # check if newly added params are in the args namespace
        # when running old generations
        args_dict = vars(args)
        if not args_dict.get("use_position_prf"):
            args.use_position_prf = False
        if not args_dict.get("code_length"):
            args.code_length = args.message_length
        if not args_dict.get("use_fixed_position"):
            args.use_fixed_position = False
        ###########################################################################
        # Extract the seeding scheme fine grained parameters
        ###########################################################################
        from utils.evaluation import scheme_hparam_extractor

        args.__dict__.update(scheme_hparam_extractor(args.seeding_scheme))

        print(f"seeding_scheme: {args.seeding_scheme}")
        print(f"prf_type: {args.prf_type}")
        print(f"anchored: {args.anchored}")
        print(f"context_width: {args.context_width}")
        print(f"self_salt: {args.self_salt}")

        ###########################################################################
        # Concat logic for multiple generations
        ###########################################################################

        if args.concat_rows != 0:
            assert isinstance(args.concat_rows, int), f"Invalid concat_rows arg: {args.concat_rows}. "

            # set to all rows if -1
            if args.concat_rows == -1:
                args.concat_rows = len(gen_table_ds)

            if args.shuffle_before_concat:
                print(f"Shuffling the gen table before concatenating every {args.concat_rows} rows...")
                gen_table_ds = gen_table_ds.shuffle()

            print(f"Concatenating every {args.concat_rows} rows of the gen table...")

            # we concat all cols in OUTPUT_TEXT_COLUMN_NAMES
            # and update the length col to reflect the new length
            # which means we need to tokenize the new text temporarily
            # to get the new length

            tokenizer = load_tokenizer(args)

            concat_partial = partial(concat_rows, tokenizer=tokenizer, args=args)

            # manually write a batch loop bc hf doesn't support returning fewer rows than input
            concatenated_rows = []
            for i in tqdm(range(0, len(gen_table_ds), args.concat_rows)):
                batch = gen_table_ds[i : i + args.concat_rows]
                concatenated_rows.append(concat_partial(batch))
            gen_table_concated_ds = Dataset.from_list(concatenated_rows)

            # overwrite the args.max_new_tokens to reflect the implicit new target length T
            # which is concat_rows * max_new_tokens
            args.max_new_tokens = args.concat_rows * args.max_new_tokens

            # write the dataset out in the same filename as the original
            # but check that the input dir is different from the output dir
            assert (
                input_dir != output_dir
            ), f"Input dir and output dir must be different to write out the result of concat rows."

            if loaded_attacked:
                concat_meta_path = f"{output_dir}/gen_table_attacked_meta.json"
                concat_gen_table_path = f"{output_dir}/gen_table_attacked.jsonl"
            else:
                concat_meta_path = f"{output_dir}/gen_table_meta.json"
                concat_gen_table_path = f"{output_dir}/gen_table.jsonl"

            write_json(args.__dict__, concat_meta_path, indent=4)
            gen_table_concated_lst = [ex for ex in gen_table_concated_ds]
            write_jsonlines(gen_table_concated_lst, concat_gen_table_path)
        else:
            gen_table_concated_ds = gen_table_ds

        ###########################################################################
        # Additional args setup
        ###########################################################################
        # if target_T is not specified, use max_new_tokens (which will be in the reloaded gen metadata)
        # and potentially overwritten by the concat logic above
        if args.target_T == 0:
            args.target_T = args.max_new_tokens

        # storing slurm info to allow auditing logfiles
        # note this is set after the metadata check to ignore overwriting
        args.SLURM_JOB_ID = os.getenv("SLURM_JOB_ID")
        args.SLURM_ARRAY_JOB_ID = os.getenv("SLURM_ARRAY_JOB_ID")
        args.SLURM_ARRAY_TASK_ID = os.getenv("SLURM_ARRAY_TASK_ID")

        ###########################################################################
        # Start logging, we wait to do this until after loading the generations
        # so that we can log the args used to generate them unioned with the
        # cmdline args
        ###########################################################################
        if args.wandb:
            # start a new wandb run to track this experiment, will send data to it
            run = wandb.init(
                # set the wandb project where this run will be logged
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=f"{run_name}_ppl_eval",
                # track hyperparameters and run metadata
                config=args,
                tags=args.wandb_tags,
            )

        ###########################################################################
        # Perplexity (PPL) evaluation
        # NOTE: basically requires a model on gpu, or is extremely slow
        ###########################################################################
        # Load the oracle model for PPL measurement
        oracle_model, oracle_tokenizer, _ = load_oracle_model(args)

        # construct the collator
        data_collator = DataCollatorWithPadding(
            tokenizer=oracle_tokenizer, padding=True, pad_to_multiple_of=8
        )

        # construct fluency/ppl partial
        evaluate_ppl_partial = partial(
            evaluate_ppl,
            oracle_model_name=args.oracle_model_name_or_path,
            oracle_model=oracle_model,
            oracle_tokenizer=oracle_tokenizer,
            data_collator=data_collator,
        )

        print(f"Computing metrics on model generations: {gen_table_concated_ds}")

        gen_table_w_ppl_ds = gen_table_concated_ds.map(
            evaluate_ppl_partial,
            batched=True,
            batch_size=args.ppl_batch_size,
            load_from_cache_file=False,
            keep_in_memory=True,
        )

        ###########################################################################
        # P-SP evaluation
        ###########################################################################

        if "p-sp" in args.evaluation_metrics:
            print(f"Loading the P-SP model and computing P-SP")
            gen_table_w_p_sp_ds = compute_p_sp(gen_table_w_ppl_ds)
        else:
            gen_table_w_p_sp_ds = gen_table_w_ppl_ds

        ###########################################################################
        # Write the final dataset out to disk in jsonl format
        # with the metrics added
        ###########################################################################

        # last applied metric, NOTE which will of course change as more are added
        gen_table_w_metrics_ds = gen_table_w_p_sp_ds

        # write the metadata file, which is a union of the previous metadata
        # and the current cmdline args
        write_json(args.__dict__, metrics_meta_path, indent=4)

        gen_table_w_metrics_lst = [ex for ex in gen_table_w_metrics_ds]
        write_jsonlines(gen_table_w_metrics_lst, gen_table_w_metrics_path)

        ###########################################################################
        # Log the metric series to wandb
        ###########################################################################
        # log the metrics to wandb
        if args.wandb:
            # find cols that should be logged in a table
            tabular_column_types = ["string", "bool"]
            tabular_column_names = [
                name
                for name, _ in filter(
                    lambda tup: tup[1].dtype in tabular_column_types,
                    gen_table_w_metrics_ds.features.items(),
                )
            ]
            # the rest should be logged as series
            series_column_names = [
                name
                for name, _ in filter(
                    lambda tup: tup[1].dtype not in tabular_column_types,
                    gen_table_w_metrics_ds.features.items(),
                )
            ]

            for metric_name in series_column_names:
                # summarize series metrics as mean by default
                wandb.define_metric(metric_name, summary="mean")

            if args.log_raw_series:
                # log the raw series
                for example in tqdm(
                    gen_table_w_metrics_ds.remove_columns(tabular_column_names),
                    desc="Logging series metrics to wandb",
                ):
                    run.log(example)

            if args.log_raw_tabular:
                # log the raw tabular data
                # but also include the dataset index as a column
                series_column_names.remove("idx")
                table = wandb.Table(
                    dataframe=gen_table_w_metrics_ds.remove_columns(series_column_names).to_pandas()
                )
                run.log({"output_table": table})

            ###########################################################################
            # Filter rows, then log means to wandb
            ###########################################################################
            args.lower_tolerance_T = min(args.lower_tolerance_T, args.target_T)
            assert (
                args.target_T - args.lower_tolerance_T
            ) >= 0, "target_T - lower_tolerance_T must be >= 0"

            target_T = args.target_T
            lower_tolerance = args.lower_tolerance_T
            upper_tolerance = args.upper_tolerance_T
            filtered_table = gen_table_w_metrics_ds.to_pandas()  # explictly convert lists
            for col in args.filter_by_columns:
                length_col_name = infer_length_column(col, filtered_table, args=args)
                filtered_table = filter_text_col_length(
                    filtered_table,
                    text_col_name=length_col_name,
                    count_suffix="",
                    upper_T=target_T + upper_tolerance,
                    lower_T=target_T - lower_tolerance,
                )

            # Save filtered mean values:
            for metric_name in series_column_names:
                filtered_name = f"filtered_{metric_name}"
                try:
                    run.summary[f"{filtered_name}_mean"] = filtered_table[metric_name].mean()
                    run.summary[f"{filtered_name}_std"] = filtered_table[metric_name].std()
                except TypeError:
                    two_dim_mean = filtered_table[metric_name].apply(np.mean).mean()
            run.finish()

    return




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computing ppl for multiple files to reduce ppl")
    parser.add_argument(
        "--evaluation_metrics",
        type=str,
        default="ppl",
        help="Computing ppl",
    )
    parser.add_argument(
        "--compute_scores_at_T",
        type=str2bool,
        default=True,
        help="Whether to compute (applicable) metrics at each T index in the output/text columns.",
    )
    parser.add_argument(
        "--overwrite_args",
        type=str2bool,
        default=False,
        help="Whether to overwrite the shared args in the metadata file with the current, runtime args.",
    )
    parser.add_argument(
        "--oracle_model_name_or_path",
        type=str,
        default="facebook/opt-6.7b",
        help="Oracle model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=None,
        help=(
            "Whether to run model (for ppl) in float16 precsion, note, will overwrite error as a reminder that "
            "generation was run in other mode, even though there's no hard requirement that these match."
        ),
    )
    parser.add_argument(
        "--ppl_batch_size",
        type=int,
        default=1,
        help="Batch size for ppl eval.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=Union[str, NoneType],
        default=None,
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=Union[float, NoneType],
        default=None,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(
        "--normalizers",
        type=Union[str, NoneType],
        default=None,
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_ngrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--return_green_token_mask",
        type=str2bool,
        default=True,
        help="Whether to return the mask marking which tokens are green from the watermark detector.",
    )
    parser.add_argument(
        "--window_settings",
        type=str,
        default="20,40,max",  # can also be "20" or "20,40,max"
        help="Comma separated list of window sizes to use for watermark detection. Only used if 'windowed-z-score' is in the evaluation metrics list.",
    )
    parser.add_argument(
        "--run_len_chisqrd_variant",
        type=str,
        default="F_succ_T_runs",
        choices=["F_succ_T_runs", "T_and_F_runs"],
        help="The variant of the run length test to use for watermark detection.",
    )
    parser.add_argument(
        "--run_len_chisqrd_bin_spec",
        type=str,
        default="max_plus_1",
        choices=["max", "max_plus_1"],
        help="The binning specification to use for the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_mask_zeros",
        type=str2bool,
        default=True,
        help="Whether to mask zeros in the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_mask_leading_bins",
        type=int,
        default=0,
        help="The number of leading bins to mask in the run length test.",
    )
    parser.add_argument(
        "--run_len_chisqrd_lambda",
        type=str,
        default="pearson",
        choices=["pearson", "g_test", "cressie_read"],
        help="The lambda_ param to use for the run length test.",
    )
    parser.add_argument(
        "--retrieval_technique",
        type=str,
        default="bm25",
        choices=["bm25", "sim"],
        help="The retrieval technique to use for retrieval detection.",
    )
    parser.add_argument(
        "--retrieval_db_column",
        type=str,
        default="no_wm_output",
        choices=["w_wm_output", "no_wm_output"],
        help="The column to populate the db/index with use for retrieval detection.",
    )
    parser.add_argument(
        "--retrieval_db_load_all_prefixes",
        type=str2bool,
        default=False,
        help="Whether to load all prefixes into the retrieval db, or just the longest for each unique entry.",
    )
    parser.add_argument(
        "--roc_test_stat",
        type=str,
        default="all",
        help="The comma separated list of test statistics to use for the ROC-AUC metric.",
    )
    parser.add_argument(
        "--target_T",
        type=int,
        default=0,
        help="The target generation length to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--lower_tolerance_T",
        type=int,
        default=25,
        help="The lower tolerance to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--upper_tolerance_T",
        type=int,
        default=25,
        help="The upper tolerance to use when dropping rows before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--filter_by_columns",
        type=str,
        default="all",
        help="The comma separated list of columns to filter by before ROC-AUC evaluation.",
    )
    parser.add_argument(
        "--wandb",
        type=str2bool,
        default=False,
        help="Whether to log to wandb.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="lm-watermarking",
        help="The name of the wandb project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="banga",
        help="The wandb entity/user for the project.",
    )
    parser.add_argument(
        "--wandb_tags",
        type=str,
        default="",
        help="The comma separated list of tags to add to the wandb run.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="",
        help="The unique name for the run.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./input",
        help="The directory containing the input files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help=(
            "The directory in which to write out the dataset after adding the metrics. "
            "If not specified, will use the input_dir. Note, if the output_dir already "
            "contains the metric-enriched file, it will be overwritten :/"
        ),
    )
    parser.add_argument(
        "--overwrite_output_file",
        type=str2bool,
        default=False,
        help="Whether to overwrite the output file if it already exists.",
    )
    parser.add_argument(
        "--limit_rows",
        type=int,
        default=-1,
        help="The number of rows to limit the dataset to. Useful for debugging.",
    )
    parser.add_argument(
        "--concat_rows",
        type=int,
        default=0,
        help="The number of rows to concatenate into a single row. Result is a mangled dataset, be careful",
    )
    parser.add_argument(
        "--shuffle_before_concat",
        type=str2bool,
        default=False,
        help="Whether to shuffle the dataset before concatenating rows.",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=None,
        help="Whether to verbosely print things here and there.",
    )
    parser.add_argument(
        "--log_raw_series",
        type=str2bool,
        default=True,
        help="Whether to log the raw series metric data to wandb.",
    )
    parser.add_argument(
        "--log_raw_tabular",
        type=str2bool,
        default=True,
        help="Whether to log the raw tabular metric data to wandb.",
    )
    parser.add_argument(
        "--debug",
        type=str2bool,
        default=False
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True
    )
    args = parser.parse_args()

    ###########################################################################
    # Argument validation and conditional setting
    ###########################################################################
    run_names = args.run_name.strip().split(",")
    args.run_name = run_names
    input_dir = [os.path.join(args.input_dir, r) for r in run_names]
    path_exist = [os.path.exists(x) for x in input_dir]
    for pe, input_d in zip(path_exist, input_dir):
        if not pe:
            print(input_d)
    assert all(path_exist), f"Above paths do not exist"

    args.input_dir = input_dir
    output_dir = [r + "_eval_ppl" for r in run_names]
    args.output_dir = output_dir

    # check limit_rows
    assert (args.limit_rows == -1) or (
        (args.limit_rows > 0) and isinstance(args.limit_rows, int)
    ), "limit_rows must be -1 or > 0"

    # convert normalizers to list
    if args.normalizers:
        args.normalizers = args.normalizers.split(",")
    else:
        args.normalizers = []

    # convert roc_test_stat to list
    args.roc_test_stat = args.roc_test_stat.split(",")

    if args.roc_test_stat == ["all"]:
        args.roc_test_stat = ROC_TEST_STAT_SUFFIXES

    # convert filter_by_columns to list
    args.filter_by_columns = args.filter_by_columns.split(",")
    if args.filter_by_columns == ["all"]:
        args.filter_by_columns = FILTER_BY_COLUMNS

    # split wandb tags
    if args.wandb_tags != "":
        args.wandb_tags = args.wandb_tags.split(",")
    else:
        args.wandb_tags = []

    # split window settings
    args.window_settings = args.window_settings.split(",")

    main(args)
