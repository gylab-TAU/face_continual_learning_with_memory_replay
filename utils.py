import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.nn import functional as F
from Memory import Memory
import pandas as pd
from PIL import Image
import os
from typing import List, Tuple
import Baby
import glob
import verification_evaluation
import random
import itertools
import math
from typing import List, Tuple, Optional, Union
import csv
from itertools import combinations


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_model_from_checkpoint(checkpoint_filename, model, optimizer):
    checkpoint = torch.load(checkpoint_filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    print(f"loaded checkpoint {checkpoint_filename}, best epoch: {best_epoch}, best_acc: {best_acc}")

    # Move model and optimizer to the specified device
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    return model, optimizer, best_epoch, best_acc


def prepare_dataset_links(dataset_link_path, dataset_roots, subset_classes):
    # make a directory for the new root
    os.makedirs(dataset_link_path, exist_ok=True)
    # first delete all previous links
    for cur_dir in os.listdir(dataset_link_path):
        if os.path.islink(os.path.join(dataset_link_path,cur_dir)):
            os.unlink(os.path.join(dataset_link_path,cur_dir))
    # now create links for class directories
    for i in range(len(dataset_roots)):
        ds = dataset_roots[i]
        classes = subset_classes[i]
        for cls_dir in classes:
            os.symlink(os.path.join(ds,cls_dir),os.path.join(dataset_link_path,cls_dir))
            print(f"created link {os.path.join(dataset_link_path,cls_dir)}")

    return dataset_link_path


def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit to allow the plots to update


def show_examples(data_subset, labels, class_names):
    # Choose the number of random samples to display
    num_samples_to_display = 15
    num_rows = 3
    num_cols = 5

    # Randomly select some samples
    random_indices = np.random.choice(len(data_subset), num_samples_to_display, replace=False)

    # Display the randomly selected samples with modified predicted labels
    plt.figure(figsize=(15, 9))

    for i, idx in enumerate(random_indices):
        sample, label = data_subset[idx]
        sample = sample.to(device).unsqueeze(0)

        # Display the original image with true label and modified predicted label
        plt.subplot(num_rows, num_cols, i + 1)
        plt.axis('off')
        plt.title(f'True: {class_names[label]}\nPredicted: {class_names[labels[idx].item()]}', fontsize=8)
        image = sample.cpu().data[0].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        plt.imshow(image)

    plt.tight_layout()
    plt.show()


def evaluate_accuracy(model, dataloader, memory, model_type=None):
    predicted_classes = []
    true_classes = []
    per_class_correct = {cls: 0 for cls in memory.memory.keys()}
    per_class_total = {cls: 0 for cls in memory.memory.keys()}

    if hasattr(dataloader.dataset, 'dataset'):
        class_to_idx = dataloader.dataset.dataset.dataset.class_to_idx
    else:
        class_to_idx = dataloader.dataset.class_to_idx

    model.to(device)
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            if model_type == 'autoencoder':
                output = model[0](inputs)
            else:
                output = model(inputs)

            embeddings = output.cpu().numpy()

            for embedding, label in zip(embeddings, labels.numpy()):
                # memory keys are class names
                predicted_class, _, _ = memory.get_predicted_class(embedding)
                predicted_classes.append(predicted_class)
                # we need to get the correct class name, by label
                class_names_for_label = [i for i in class_to_idx if class_to_idx[i] == label]
                if len(class_names_for_label) != 1:
                    print('got bad number of class_names_for_label')
                    return
                class_name = class_names_for_label[0]
                true_classes.append(class_name)

                per_class_total[class_name] += 1
                if predicted_class == class_name:
                    per_class_correct[class_name] += 1

    per_class_accuracy = {cls: per_class_correct[cls] / per_class_total[cls] for cls in memory.memory.keys()}
    pred_classes_only = [pl for pl in predicted_classes]
    overall_accuracy = accuracy_score(true_classes, pred_classes_only)

    return per_class_accuracy, overall_accuracy, true_classes, predicted_classes



def generate_random_index_lists(index_range=20, num_selected=8, num_lists=10, seed=42):
    """
    Generate a list of unique lists, each containing randomly selected indices.

    Parameters:
    - index_range (int): Total range of indices (0 to index_range - 1). Default is 20.
    - num_selected (int): Number of indices to randomly select from the range. Default is 8.
    - num_lists (int): Number of such unique lists to generate. Default is 10.
    - seed (int or None): Random seed for reproducibility. Default is 42. Set to None for full randomness.

    Returns:
    - List[List[int]]: A list containing `num_lists` unique sublists of randomly selected and shuffled indices.
    """
    if num_selected > index_range:
        raise ValueError("num_selected cannot be greater than index_range")

    if seed is not None:
        random.seed(seed)

    all_indices = list(range(index_range))
    result = set()

    attempt_limit = 10000
    attempts = 0

    while len(result) < num_lists and attempts < attempt_limit:
        selected = random.sample(all_indices, num_selected)
        random.shuffle(selected)
        frozen = tuple(selected)
        result.add(frozen)
        attempts += 1

    if len(result) < num_lists:
        raise RuntimeError(f"Could not generate {num_lists} unique lists after {attempt_limit} attempts")

    return [list(t) for t in result]


def generate_progressive_pairs(index_list: List[int], seed: int = None) -> List[Tuple[int, int]]:
    """
    Given a list of integers, generate a list of pairs in progressive steps:
    - Step 1: first 2 elements → 1 pair
    - Step 2: first 3 elements → 3 pairs
    - ...
    - Step N: all elements → C(n, 2) pairs

    Within each step, pairs are shuffled randomly.

    Parameters:
    - index_list (List[int]): The input list of unique integers.
    - seed (int or None): Random seed for reproducibility. Default is None.

    Returns:
    - List[Tuple[int, int]]: A list of all pairs from all steps, randomized within each step.
    """
    if len(index_list) < 2:
        return []

    if seed is not None:
        random.seed(seed)

    all_pairs = []
    step_indices = []

    for i in range(2, len(index_list) + 1):
        current_subset = index_list[:i]
        step_pairs = list(itertools.combinations(current_subset, 2))
        random.shuffle(step_pairs)
        all_pairs.extend(step_pairs)
        # add it again
        random.shuffle(step_pairs)
        all_pairs.extend(step_pairs)

        step_indices.append(len(all_pairs))

    return all_pairs, step_indices

def generate_progressive_inclusion_lists(index_list: List[int], seed: int = None) -> Tuple[List[List[int]], List[int]]:
    """
    Build a list of index lists using progressive inclusion logic,
    where each list has at most 4 indices. Tracks step start indices.

    Parameters:
    - index_list (List[int]): Must have at least 3 elements.
    - seed (int or None): For reproducible shuffling.

    Returns:
    - Tuple:
        - List[List[int]]: All valid sublists, step by step, shuffled per step.
        - List[int]: Start index of each step in the combined list.
    """
    if len(index_list) < 3:
        raise ValueError("index_list must contain at least 3 elements")

    if seed is not None:
        random.seed(seed)

    base = index_list[:3]
    result = [base[:]]  # Step 1
    step_indices = [0]  # First step starts at 0
    current_lists = [base[:]]

    for i in range(3, len(index_list)):
        new_index = index_list[i]
        next_lists = []

        for lst in current_lists:
            # Keep the list as-is
            next_lists.append(lst[:])

            # Add the new index if size allows
            if len(lst) < 4:
                next_lists.append(lst + [new_index])

        # Only keep lists with ≤ 4 elements
        next_lists = [lst for lst in next_lists if len(lst) <= 4]

        random.shuffle(next_lists)
        step_indices.append(len(result))  # mark start of this step
        result.extend(next_lists)
        current_lists = next_lists

    return result, step_indices


def generate_protocol_1_data():
    # 10 repetitions, each with 10 identities out of 20.
    lists = generate_random_index_lists(
        index_range=20,
        num_selected=10,
        num_lists=10,
        seed=42
    )

    with open('protocol_1_experiment_data.txt', 'w') as f:

        for run_index, lst in enumerate(lists):

            ae_list, ae_step_ends = generate_progressive_pairs(
                lst,
                S=math.inf,
                duplicates=0,
                coverage=True,
                seed=123
            )

            embd_list, embd_step_ends = generate_progressive_pairs(
                lst,
                S=math.inf,
                duplicates=0,
                coverage=True,
                seed=123
            )

            cl_test_indices = [
                step_end - 1
                for step_end in embd_step_ends
            ]

            # Map Embedder stages to the corresponding AE stages.
            dependencies = generate_protocol_1_dependencies(
                ae_step_ends,
                embd_step_ends
            )

            if len(dependencies) != len(embd_list):
                raise ValueError(
                    f"Protocol 1 run {run_index}: dependency count "
                    f"({len(dependencies)}) does not match Embedder "
                    f"steps ({len(embd_list)})."
                )

            batch_list = [
                lst[:num_identities]
                for num_identities in range(2, len(lst) + 1)
            ]

            batch_test_indices = [
                step for step in range(len(batch_list))
            ]

            f.write(f'run_{run_index}\n')
            f.write(f'identity_list: {lst}\n')
            f.write(f'ae_list: {ae_list}\n')
            f.write(f'embd_list: {embd_list}\n')
            f.write(f'embd_ae_dependencies: {dependencies}\n')
            f.write(f'cl_test_indices: {cl_test_indices}\n')
            f.write(f'batch_list: {batch_list}\n')
            f.write(f'batch_test_indices: {batch_test_indices}\n')            
            f.write('\n')


def generate_protocol_2_data():
    # 10 repetitions, each with 8 identities out of 20.
    lists = generate_random_index_lists(
        index_range=20,
        num_selected=8,
        num_lists=10,
        seed=123
    )

    with open('protocol_2_experiment_data.txt', 'w') as f:

        for run_index, lst in enumerate(lists):

            # AE training schedule.
            ae_list, ae_step_ends = generate_progressive_pairs(
                lst,
                S=math.inf,
                duplicates=1,
                coverage=True,
                seed=123
            )

            # First three Embedder steps: pairwise training
            # without requiring AE-generated images.
            embd_list = [
                [lst[0], lst[1]],
                [lst[0], lst[2]],
                [lst[1], lst[2]]
            ]

            # Progressive Embedder inclusion schedule.
            embedder_steps_list, inclusion_step_starts = (
                generate_progressive_inclusion_lists(
                    lst,
                    seed=123
                )
            )

            embd_list.extend(embedder_steps_list)

            inclusion_step_ends = (
                inclusion_step_starts[1:]
                + [len(embedder_steps_list)]
            )

            cl_test_indices = [0, 2] + [
                3 + step_end - 1
                for step_end in inclusion_step_ends
            ]

            # Generate AE dependencies from the schedule boundaries.
            dependencies = generate_protocol_2_dependencies(
                ae_step_ends=ae_step_ends,
                inclusion_step_starts=inclusion_step_starts,
                inclusion_list_length=len(embedder_steps_list)
            )

            if len(dependencies) != len(embd_list):
                raise ValueError(
                    f"Protocol 2 run {run_index}: dependency count "
                    f"({len(dependencies)}) does not match Embedder "
                    f"steps ({len(embd_list)})."
                )

            batch_list = [
                lst[:num_identities]
                for num_identities in range(2, len(lst) + 1)
            ]

            batch_test_indices = [
                step for step in range(len(batch_list))
            ]

            f.write(f'run_{run_index}\n')
            f.write(f'identity_list: {lst}\n')
            f.write(f'ae_list: {ae_list}\n')
            f.write(f'embd_list: {embd_list}\n')
            f.write(f'embd_ae_dependencies: {dependencies}\n')
            f.write(f'cl_test_indices: {cl_test_indices}\n')
            f.write(f'batch_list: {batch_list}\n')
            f.write(f'batch_test_indices: {batch_test_indices}\n')            
            f.write('\n')


def generate_progressive_pairs(
    index_list: List[int],
    S: Union[int, float] = math.inf,
    duplicates: int = 1,
    coverage: bool = False,
    seed: Optional[int] = None,
    output_file: Optional[str] = None,
    return_step_indices: bool = True,
):
    """
    Generate pairs from progressively growing prefixes of index_list.

    At each step, all pairs are generated from the first 2, 3, ...,
    len(index_list) elements. Up to S unique pairs are selected,
    subject to the coverage option.

    Parameters
    ----------
    index_list : List[int]
        Unique identity indices in the desired order.

    S : int or float, default=math.inf
        Maximum number of unique pairs selected per step.
        Use math.inf to include all possible pairs.
        If coverage=True, coverage may require exceeding S.

    duplicates : int, default=1
        Number of EXTRA copies of each selected pair within a step.
        0 means no extra copies (each pair appears once).
        1 means one extra copy (each pair appears twice).
        2 means two extra copies (each pair appears three times).

    coverage : bool, default=False
        If True, ensure every identity in the current step appears
        in at least one selected pair.

    seed : int or None, default=None
        Random seed for reproducibility.

    output_file : str or None, default=None
        If provided, write one pair per line to this file.

    return_step_indices : bool, default=True
        If True, return (all_pairs, step_indices).
        Each step index is the cumulative number of pairs after
        completing that step, including duplicate copies.
        If False, return only all_pairs.

    Returns
    -------
    all_pairs : list[tuple[int, int]]
        Generated pairs, including repetitions.

    step_indices : list[int], optional
        Cumulative end positions for each progressive step.
    """
    if len(index_list) != len(set(index_list)):
        raise ValueError("index_list must contain unique integers.")

    if not isinstance(duplicates, int) or isinstance(duplicates, bool):
        raise ValueError("duplicates must be a non-negative integer.")
    if duplicates < 0:
        raise ValueError("duplicates must be >= 0.")

    if S != math.inf:
        if not isinstance(S, (int, float)) or isinstance(S, bool):
            raise ValueError("S must be a non-negative integer or math.inf.")
        if not math.isfinite(S) or S < 0 or not S.is_integer():
            # Integers do not have is_integer(); handle them below instead.
            if not (isinstance(S, int) and S >= 0):
                raise ValueError("S must be a non-negative integer or math.inf.")
        S = int(S)

    rng = random.Random(seed)
    all_pairs = []
    step_indices = []

    for i in range(2, len(index_list) + 1):
        current_subset = index_list[:i]
        candidates = list(itertools.combinations(current_subset, 2))
        rng.shuffle(candidates)

        if coverage:
            # Select pairs so that every identity is represented.
            selected = []
            covered = set()

            for identity in current_subset:
                if identity not in covered:
                    pair = next(
                        pair for pair in candidates
                        if identity in pair and pair not in selected
                    )
                    selected.append(pair)
                    covered.update(pair)

            # Add random pairs until reaching S, if necessary.
            # Coverage takes priority if its pairs already exceed S.
            if S == math.inf:
                target = len(candidates)
            else:
                target = max(int(S), len(selected))

            remaining = [pair for pair in candidates if pair not in selected]
            selected.extend(remaining[:max(0, target - len(selected))])
        else:
            if S == math.inf:
                selected = candidates
            else:
                selected = candidates[:min(int(S), len(candidates))]

        # duplicates counts EXTRA copies, so total repetitions = duplicates + 1.
        for pair in selected:
            all_pairs.extend([pair] * (duplicates + 1))

        step_indices.append(len(all_pairs))

    if output_file is not None:
        with open(output_file, "w", encoding="utf-8") as file:
            for pair in all_pairs:
                file.write(f"{pair}\n")

    if return_step_indices:
        return all_pairs, step_indices

    return all_pairs

def generate_protocol_1_dependencies(ae_step_ends, embd_step_ends):
    """
    Map each Embedder step to the corresponding AE stage.

    Each dependency is the zero-based index of the final AE
    sub-step in that progressive stage.
    """
    if len(ae_step_ends) != len(embd_step_ends):
        raise ValueError(
            "AE and Embedder schedules must have the same "
            "number of progressive stages."
        )

    dependencies = []
    previous_embd_end = 0

    for ae_end, embd_end in zip(ae_step_ends, embd_step_ends):
        # Number of Embedder steps in this progressive stage.
        stage_size = embd_end - previous_embd_end

        # All Embedder steps in this stage depend on the
        # final AE sub-step of the corresponding AE stage.
        dependencies.extend([ae_end - 1] * stage_size)

        previous_embd_end = embd_end

    return dependencies



def generate_protocol_2_dependencies(
    ae_step_ends,
    inclusion_step_starts,
    inclusion_list_length,
):
    """
    Generate Protocol 2 dependencies.

    The first three Embedder steps use real images only and have
    no AE dependency. Subsequent inclusion stages depend on the
    final AE sub-step for the corresponding number of identities.
    """
    dependencies = [None, None, None]

    for stage_index, stage_start in enumerate(inclusion_step_starts):
        if stage_index + 1 < len(inclusion_step_starts):
            stage_end = inclusion_step_starts[stage_index + 1]
        else:
            stage_end = inclusion_list_length

        # The first inclusion stage contains 3 identities.
        # Each subsequent stage introduces one more identity.
        num_identities = stage_index + 3

        # AE step ends are cumulative counts, so subtract 1
        # to get the zero-based index of the final AE sub-step.
        ae_dependency = ae_step_ends[num_identities - 2] - 1

        dependencies.extend(
            [ae_dependency] * (stage_end - stage_start)
        )

    return dependencies


def calc_generated_images_mse(
    dir_list,
    run_index,
    all_identities,
    base_checkpoints_path,
    dataset_link_path_train,
    real_root,
):
    """
    Calculate reconstruction MSE for generated images at selected
    autoencoder checkpoints.

    Parameters
    ----------
    dir_list : list[int]
        Zero-based AE sub-step indices to evaluate.

    run_index : int
        Protocol run number.

    all_identities : list[str]
        Global list of identity names. Identity indices stored in the
        AE memory are mapped to these names.

    base_checkpoints_path : str
        Path containing the autoencoder sub-step directories.

    dataset_link_path_train : str
        Path to the training dataset links used by the autoencoder.

    real_root : str
        Path to the real training gallery used to prepare dataset links.

    Returns
    -------
    None
        Prints the mean and standard deviation of reconstruction MSE
        for each identity at each selected checkpoint.
    """

    cl_baby = Baby()

    for cur_index in dir_list:

        print(
            f'calc_generated_images_mse, '
            f'run_index: {run_index}, '
            f'cur_index: {cur_index}'
        )

        # Identify the directory for this AE checkpoint.
        checkpoint_path = glob.glob(
            os.path.join(
                base_checkpoints_path,
                f'sub_step_{cur_index}'
            )
        )

        if len(checkpoint_path) == 0:
            print('could not find dir for step')
            return

        checkpoint_path = checkpoint_path[0]

        print(f'directory for step: {checkpoint_path}')

        # Load the AE memory saved at this checkpoint.
        print(f'loading previous ae memory from {checkpoint_path}')
        cl_baby.load_from_disk(checkpoint_path)

        ae_memory = cl_baby.ae_memory

        print('identities in ae memory:')

        train_classes = []

        for key, _ in ae_memory.memory.items():
            train_classes.append(key)

        print(f'train_classes: {train_classes}')

        # Convert the identity indices in AE memory to identity names.
        identity_names = [
            all_identities[class_index]
            for class_index in train_classes
        ]

        print(f'identity names: {identity_names}')

        # Prepare links to the real images corresponding to the
        # identities currently represented in AE memory.
        dataset_link_path = prepare_dataset_links(
            dataset_link_path_train,
            [real_root],
            [train_classes]
        )

        # Load the current autoencoder checkpoint.
        cl_baby.train_ae2(
            dataset_link_path,
            checkpoint_path,
            read_from_disk=True,
            resume_dir=checkpoint_path,
            get_from_checkpoint=True
        )

        # Calculate reconstruction MSE for the current identities.
        mse_per_class = cl_baby.evaluate_mse(dataset_link_path)

        for cls, mse_vec in mse_per_class.items():
            mean_mse = np.mean(mse_vec)
            mse_std = np.std(mse_vec)

            print(
                f'class: {cls} '
                f'({all_identities[cls]}) '
                f'mean_mse: {mean_mse:.4f}, '
                f'mse_std: {mse_std:.4f}'
            )

    return


def generate_face_verification_pairs_all(
        base_path,
        cur_train_ids,
        output_file,
        num_same_pairs=100,
        num_diff_pairs=100,
        seed=42
):
    """
    Generate a balanced face-verification pair dataset.

    train_pair = 1:
        Both identities in the pair are trained.
        For same-identity pairs, the identity is trained.

    train_pair = 0:
        At least one identity is not trained.
        For same-identity pairs, the identity is not trained.

    Same-identity and different-identity pairs are balanced
    separately between train_pair=1 and train_pair=0.

    Output CSV columns (no header):
        img1_path, img2_path, same_identity, train_pair
    """

    rng = random.Random(seed)
    cur_train_ids = set(cur_train_ids)

    if num_same_pairs < 0 or num_diff_pairs < 0:
        raise ValueError("Pair counts must be non-negative.")

    # -------------------------------------------------------
    # Collect images per identity
    # -------------------------------------------------------

    identity_images = {}

    for identity in sorted(os.listdir(base_path)):
        identity_path = os.path.join(base_path, identity)

        if not os.path.isdir(identity_path):
            continue

        images = [
            os.path.join(identity_path, fname)
            for fname in sorted(os.listdir(identity_path))
            if os.path.isfile(os.path.join(identity_path, fname))
        ]

        if len(images) >= 2:
            identity_images[identity] = images

    identities = list(identity_images)

    trained_ids = [
        identity for identity in identities
        if identity in cur_train_ids
    ]

    untrained_ids = [
        identity for identity in identities
        if identity not in cur_train_ids
    ]

    if len(identities) < 2:
        raise ValueError("Need at least two identities with 2+ images.")

    # -------------------------------------------------------
    # Split requested counts as evenly as possible
    # -------------------------------------------------------

    def balanced_counts(total):
        # If odd, the two categories differ by one pair.
        return total // 2, total - total // 2

    n_same_train, n_same_untrained = balanced_counts(
        num_same_pairs
    )

    n_diff_train, n_diff_untrained = balanced_counts(
        num_diff_pairs
    )

    # -------------------------------------------------------
    # Validate that each requested category is possible
    # -------------------------------------------------------

    if n_same_train > 0 and not trained_ids:
        raise ValueError(
            "Cannot generate trained same-identity pairs: "
            "no trained identities have 2+ images."
        )

    if n_same_untrained > 0 and not untrained_ids:
        raise ValueError(
            "Cannot generate untrained same-identity pairs: "
            "no untrained identities have 2+ images."
        )

    trained_combinations = list(combinations(trained_ids, 2))

    untrained_combinations = [
        pair for pair in combinations(identities, 2)
        if not (
            pair[0] in cur_train_ids
            and pair[1] in cur_train_ids
        )
    ]

    if n_diff_train > 0 and not trained_combinations:
        raise ValueError(
            "Cannot generate trained different-identity pairs: "
            "need at least two trained identities with 2+ images."
        )

    if n_diff_untrained > 0 and not untrained_combinations:
        raise ValueError(
            "Cannot generate untrained different-identity pairs: "
            "no eligible identity combinations."
        )

    # -------------------------------------------------------
    # Generate same-identity pairs
    # -------------------------------------------------------

    same_pairs = []

    def add_same_pairs(ids, count, train_pair):
        for _ in range(count):
            identity = rng.choice(ids)
            img1, img2 = rng.sample(
                identity_images[identity], 2
            )

            same_pairs.append(
                (img1, img2, 1, train_pair)
            )

    add_same_pairs(trained_ids, n_same_train, 1)
    add_same_pairs(untrained_ids, n_same_untrained, 0)

    # -------------------------------------------------------
    # Generate different-identity pairs
    # -------------------------------------------------------

    diff_pairs = []

    def add_diff_pairs(identity_combinations, count, train_pair):
        for _ in range(count):
            id1, id2 = rng.choice(identity_combinations)

            img1 = rng.choice(identity_images[id1])
            img2 = rng.choice(identity_images[id2])

            diff_pairs.append(
                (img1, img2, 0, train_pair)
            )

    add_diff_pairs(
        trained_combinations, n_diff_train, 1
    )

    add_diff_pairs(
        untrained_combinations, n_diff_untrained, 0
    )

    # -------------------------------------------------------
    # Shuffle and write CSV (preserve no-header format)
    # -------------------------------------------------------

    all_pairs = same_pairs + diff_pairs
    rng.shuffle(all_pairs)

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(all_pairs)

    # -------------------------------------------------------
    # Report counts
    # -------------------------------------------------------

    print(f"Generated {len(same_pairs)} same-identity pairs:")
    print(f"  train_pair=1: {n_same_train}")
    print(f"  train_pair=0: {n_same_untrained}")

    print(f"Generated {len(diff_pairs)} different-identity pairs:")
    print(f"  train_pair=1: {n_diff_train}")
    print(f"  train_pair=0: {n_diff_untrained}")

    print(f"Saved to {output_file}")


def evaluate_verification_accuracy(model, pairs_file):

    model.to(device)
    results = verification_evaluation.evaluate_verification(model,pair_file=pairs_file)
    print(results)


