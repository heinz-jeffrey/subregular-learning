import argparse
import os, sys
from itertools import product

"""
Print language names satisfying a certain property
Args:
    --action:
        all_fst:
            lang names with an .fst file in src/fstlib/fst_format/
        all_langs:
            lang names with an .fst or with data in data_gen
            (includes complementary languages)
        data_gen_done:
            lang names for which data generation is complete
        train_done:
            lang names for which training & eval is complete
    --avoid:
        a file whose lines are language names not to print
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="data_gen_done")
    parser.add_argument("--avoid_file", type=str, default=None)
    parser.add_argument("--model_dir",  type=str, default="models")
    args = parser.parse_args()

    avoid = (
        [l.strip() for l in open(args.avoid_file, "r").readlines()]
        if args.avoid_file is not None else []
    )

    if args.action == "all_fst" or args.action == "all_langs":
        fst_names = sorted([
            filename.replace(".fst", "")
            for filename in os.listdir("src/fstlib/fst_format/")
        ])
        langs_out = sorted(f"{l}" for l in fst_names if l not in avoid)

        if args.action == "all_langs":
            small_files = [f for f in os.listdir("data_gen/Small")]
            mid_files = [f for f in os.listdir("data_gen/Mid")]
            large_files = [f for f in os.listdir("data_gen/Large")]
            langs_with_data = sorted(
                set(f.split("_")[0] for f in small_files) |
                set(f.split("_")[0] for f in mid_files) |
                set(f.split("_")[0] for f in large_files)
            )
            langs_out = sorted(set(langs_out + langs_with_data))

    elif args.action == "data_gen_done":
        small_files = [
            f
            for f in os.listdir("data_gen/Small")
            if len(open(f"data_gen/Small/{f}", "r").readlines()) == 1e3
        ]
        mid_files = [
            f
            for f in os.listdir("data_gen/Mid")
            if len(open(f"data_gen/Mid/{f}", "r").readlines()) == 1e4
        ]
        large_files = [
            f
            for f in os.listdir("data_gen/Large")
            if len(open(f"data_gen/Large/{f}", "r").readlines()) == 1e5
        ]

        langs_with_data = sorted(
            set(f.split("_")[0] for f in small_files) |
            set(f.split("_")[0] for f in mid_files) |
            set(f.split("_")[0] for f in large_files)
        )

        langs_out = []
        f_types = ["Dev", "TestSR", "TestSA", "TestLR", "TestLA", "Train"]
        for lang in langs_with_data:
            files = [f"{lang}_{f_type}.txt" for f_type in f_types]
            all_files_complete = all(
                [f in small_files for f in files] +
                [f in mid_files for f in files] +
                [f in large_files for f in files]
            )
            if all_files_complete and lang not in avoid:
                langs_out.append(lang)

    elif args.action == "train_done":
        directions = ["Uni"]
        network_types = [
            "simple",
            "gru",
            "lstm",
            "transformer"
        ]
        drops = ["NoDrop"]
        sizes = ["Small", "Mid", "Large"]

        test_types = ["SR", "SA", "LR", "LA"]
        model_complete = lambda model: all(
            f"Test{test}_eval.txt" in os.listdir(f"{args.model_dir}/{model}")
            if os.path.exists(f"{args.model_dir}/{model}") else False
            for test in test_types
        )

        langs_with_models = sorted(
            set(f.split("_")[3] for f in os.listdir(args.model_dir))
        )
        langs_out = []
        for lang in langs_with_models:
            model_grid = product(directions, network_types, drops, sizes)
            model_list = [
                f"{direction}_{network_type}_{drop}_{lang}_{size}"
                for direction, network_type, drop, size in model_grid
            ]
            if (
                all(model_complete(model) for model in model_list) and
                lang not in avoid
            ):
                langs_out.append(lang)

    with open("/dev/stdout", "w") as f:
            f.write("\n".join(langs_out))
