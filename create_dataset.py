# Copyright (c) 2024-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# HOW TO REPLICATE THE MIXTURE:
# 1) Remove duplicate in the file
# 2) Remove INT+ 0 in the bwd file on both X and y
# 3) Create the mixture with different weights
# 4) Create train-test split


import io
import math
import random
import subprocess


def remove_duplicate_in_the_file(old_file, new_file, separator):
    if separator:
        command = """
        awk '{{!seen[$0]++}}END{{for (i in seen) printf("%i|%s\\n",seen[i],i)}}' "{old_file}" | cat > "{new_file}"
        """.format(
            old_file=old_file, new_file=new_file
        )
    else:
        command = """
        awk '{{!seen[$0]++}}END{{for (i in seen) printf("%s\\n",i)}}' "{old_file}" | cat > "{new_file}"
        """.format(
            old_file=old_file, new_file=new_file
        )
    subprocess.run(command, shell=True)


def remote_int_zeros_bwd(old_file, new_file):
    to_store = []
    with open(old_file, "r") as open_file:
        for l in open_file:
            x, y = l.split("\t")
            systems = x.split("<SPECIAL_3>")[1:]
            flag = True
            for sys in systems:
                if sys.strip() == "INT+ 0":
                    flag = False
            if y.startswith("+ INT+ 0 + "):
                assert len(y) - len(y.replace("+ INT+ 0 ", "")) == len("+ INT+ 0 ")
                y = y.replace("+ INT+ 0 ", "")
            if flag:
                to_store.append(f"{x}\t{y}")
    with open(new_file, "w") as file_handler_prefix:
        for l in to_store:
            file_handler_prefix.write(l)


def create_mixture(files, weights, new_file):
    assert len(files) == len(weights)
    to_store = []
    for i, file in enumerate(files):
        with open(file, "r") as open_file:
            lines = [l for l in open_file]
        size = int(len(lines) * weights[i])
        total_size = len(lines)
        alpha = math.log(total_size - 0.5) / math.log(size)
        indices = [int(i**alpha) for i in range(1, size + 1)]
        to_store.extend(lines[i] for i in indices)

    random.shuffle(to_store)

    with open(new_file, "w") as file_handler_prefix:
        for el in to_store:
            file_handler_prefix.write(el)


def split_mixture_train_eval_test(file, vld_tst_size, valid: bool, suffix: list):
    trn_path = file + suffix[0]
    if valid:
        vld_path = file + suffix[1]
    tst_path = file + (suffix[2] if valid else suffix[1])

    with io.open(file, mode="r", encoding="utf-8") as f:
        lines = [line for line in f]
    total_size = len(lines)
    assert 2 * vld_tst_size < total_size if valid else vld_tst_size < total_size

    alpha = math.log(total_size - 0.5) / math.log(2 * vld_tst_size if valid else vld_tst_size)
    assert int((2 * vld_tst_size if valid else vld_tst_size) ** alpha) == total_size - 1
    vld_tst_indices = [int(i**alpha) for i in range(1, 2 * vld_tst_size + 1 if valid else vld_tst_size + 1)]
    if valid:
        vld_indices = set(vld_tst_indices[::2])
        tst_indices = set(vld_tst_indices[1::2])
    else:
        tst_indices = set(vld_tst_indices)
    if valid:
        assert len(vld_tst_indices) == 2 * vld_tst_size
        assert len(vld_indices) == vld_tst_size
    assert max(vld_tst_indices) == total_size - 1
    assert len(tst_indices) == vld_tst_size

    f_train = io.open(trn_path, mode="w", encoding="utf-8")
    if valid:
        f_valid = io.open(vld_path, mode="w", encoding="utf-8")
    f_test = io.open(tst_path, mode="w", encoding="utf-8")

    for i, line in enumerate(lines):
        if valid and i in vld_indices:
            f_valid.write(line)
        elif i in tst_indices:
            f_test.write(line)
        else:
            f_train.write(line)

    f_train.close()
    if valid:
        f_valid.close()
    f_test.close()


def clean_validation(old_file):
    with open(f"{old_file}.train", "r") as train_file, open(f"{old_file}.valid", "r") as valid_file, open(
        f"{old_file}.valid.final", "w"
    ) as final_valid_file, open(f"{old_file}.test", "r") as test_file, open(f"{old_file}.test.final", "w") as final_test_file:
        train_lines = {line.split("|")[1] for line in train_file}

        for line in valid_file:
            if line.split("|")[1] not in train_lines:
                final_valid_file.write(line)
        for line in test_file:
            if line.split("|")[1] not in train_lines:
                final_test_file.write(line)


def run_mixture(files, weights):
    new_file = "/path/to/your/dataset"  # amend this path
    new_file += ".".join([file.split("/")[-1] + "." + str(int(100 * weight)) for file, weight in zip(files, weights)])
    create_mixture(files, weights, new_file)
    new_file_cleaned = new_file + ".cleaned"
    remove_duplicate_in_the_file(new_file, new_file_cleaned, separator=True)
    split_mixture_train_eval_test(new_file_cleaned, 200, True, [".train", ".valid", ".test"])
    clean_validation(new_file_cleaned)
    print(new_file_cleaned)


# Warning: amend the two paths to mix the data for training
run_mixture(["/path/to/your/bwd_dataset", "/path/to/your/fwd_dataset"], [1.0, 1.0])
