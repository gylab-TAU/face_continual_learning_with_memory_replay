import argparse
import ast
import os
import time
from config import Config
import continual_learning
import utils


# ------------------------------------------------------------
# Experimental configuration
# ------------------------------------------------------------

PROTOCOL_CONFIG = {
    "protocol_1": {
        "data_file": "protocol_1_experiment_data.txt",
    },
    "protocol_2": {
        "data_file": "protocol_2_experiment_data.txt",
    },
}

TRAIN_MODES = ["CL_w_replay", "batch", "naive_CL"]



# ------------------------------------------------------------
# Experiment-data and path helpers
# ------------------------------------------------------------

def load_experiment_data(filename):
    """Load the run blocks stored in a protocol experiment-data file."""
    experiment_data = []
    current_run = None

    with open(filename, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    for line in lines:
        if line.startswith("run_"):
            if current_run is not None:
                experiment_data.append(current_run)

            current_run = {
                "run_index": int(line.split("_", 1)[1])
            }

        elif current_run is None:
            raise ValueError(
                f"Found data before the first run header in {filename}: "
                f"{line}"
            )

        elif ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if key in {
                "identity_list",
                "ae_list",
                "embd_list",
                "batch_list",
                "embd_ae_dependencies",
                "cl_test_indices",
                "batch_test_indices",
            }:
                current_run[key] = ast.literal_eval(value)

    if current_run is not None:
        experiment_data.append(current_run)

    if not experiment_data:
        raise ValueError(f"No experiment runs found in {filename}")

    return experiment_data


def write_step_train_ids(step_dir, train_ids):
    os.makedirs(step_dir, exist_ok=True)
    with open(
        os.path.join(step_dir, "step_train_ids.txt"),
        "w",
        encoding="utf-8",
    ) as file:
        file.write(str(train_ids))


def get_step_dir(run_dir, step):
    return os.path.join(run_dir, f"sub_step_{step}")


def get_previous_step_dir(run_dir, step):
    if step == 0:
        return None
    return get_step_dir(run_dir, step - 1)


def get_train_classes(train_ids):
    """Convert global identity indices into dataset class names."""
    for identity_index in train_ids:
        if not isinstance(identity_index, int):
            raise TypeError(
                f"Identity indices must be integers; got {identity_index!r}"
            )
        if not 0 <= identity_index < len(Config.all_identities):
            raise ValueError(
                f"Identity index {identity_index} is outside "
                f"0..{len(Config.all_identities) - 1}"
            )

    return [Config.all_identities[index] for index in train_ids]


# ------------------------------------------------------------
# Shared experiment dispatch
# ------------------------------------------------------------

def main(base_path, protocols):
    for protocol in protocols:
        for train_mode in TRAIN_MODES:
            run_experiment(protocol, train_mode, base_path)


def run_experiment(protocol, train_mode, base_path):
    if protocol not in PROTOCOL_CONFIG:
        raise ValueError(f"Unknown protocol: {protocol}")

    config = PROTOCOL_CONFIG[protocol]
    experiment_data = load_experiment_data(config["data_file"])

    output_path = os.path.join(base_path, protocol, train_mode)

    print("=" * 60)
    print(f"Protocol: {protocol}")
    print(f"Training mode: {train_mode}")
    print(f"Output directory: {output_path}")
    print("=" * 60)

    if train_mode == "CL_w_replay":
        run_cl_w_replay(
            base_path=output_path,
            experiment_data=experiment_data
        )

    elif train_mode == "batch":
        run_batch(
            base_path=output_path,
            experiment_data=experiment_data
        )

    elif train_mode == "naive_CL":
        run_naive_cl(
            base_path=output_path,
            experiment_data=experiment_data
        )

    else:
        raise ValueError(f"Unknown training mode: {train_mode}")



def run_cl_w_replay(
    base_path,
    experiment_data,
    max_runs=None,
    max_embedder_steps=None,
):
    """
    Run the same dependency-aware AE/embedder procedure for either protocol.

    Each run's data must contain:
      - ae_list
      - embd_list
      - embd_ae_dependencies

    Dependency values are zero-based AE sub-step indices. None means that
    the embedder step does not use AE replay.
    """
    runs = experiment_data if max_runs is None else experiment_data[:max_runs]

    for run_position, run_data in enumerate(runs):
        run_index = run_data.get("run_index", run_position)
        print(f"\nStarting run {run_index}")

        required_keys = {
            "ae_list",
            "embd_list",
            "embd_ae_dependencies",
            "cl_test_indices",
        }
        missing = required_keys - run_data.keys()

        if missing:
            raise ValueError(
                f"Run {run_index} is missing required fields: "
                f"{sorted(missing)}. Regenerate the experiment data files."
            )

        ae_list = run_data["ae_list"]
        embd_list = run_data["embd_list"]
        dependencies = run_data["embd_ae_dependencies"]
        run_test_indices = run_data["cl_test_indices"]


        if len(dependencies) != len(embd_list):
            raise ValueError(
                f"Run {run_index}: {len(dependencies)} AE dependencies "
                f"but {len(embd_list)} embedder steps."
            )

        for dependency in dependencies:
            if dependency is not None and not (
                isinstance(dependency, int) and 0 <= dependency < len(ae_list)
            ):
                raise ValueError(
                    f"Run {run_index}: invalid AE dependency {dependency!r}; "
                    f"valid indices are 0..{len(ae_list) - 1} or None."
                )

        ae_run_dir = os.path.join(
            base_path, "autoencoder", f"run_{run_index}"
        )
        embd_run_dir = os.path.join(
            base_path, "embedder", f"run_{run_index}"
        )
        os.makedirs(ae_run_dir, exist_ok=True)
        os.makedirs(embd_run_dir, exist_ok=True)

        # Index of the next AE sub-step that has not yet been trained.
        next_ae_step = 0
        steps_to_run = (
            len(embd_list)
            if max_embedder_steps is None
            else min(max_embedder_steps, len(embd_list))
        )

        for embd_step in range(steps_to_run):
            train_ids = embd_list[embd_step]
            ae_required_step = dependencies[embd_step]

            print(
                f"\nEmbedder sub-step {embd_step}: IDs={train_ids}; "
                f"required AE step={ae_required_step}"
            )

            # Train all AE steps required by this embedder step.
            while (
                ae_required_step is not None
                and next_ae_step <= ae_required_step
            ):
                if next_ae_step >= len(ae_list):
                    raise ValueError(
                        f"Run {run_index}: dependency {ae_required_step} "
                        "requires more AE steps than ae_list contains."
                    )

                ae_ids = ae_list[next_ae_step]
                ae_classes = get_train_classes(ae_ids)
                ae_step_dir = get_step_dir(ae_run_dir, next_ae_step)
                prev_ae_dir = get_previous_step_dir(
                    ae_run_dir, next_ae_step
                )

                write_step_train_ids(ae_step_dir, ae_ids)

                print(
                    f"Training AE sub-step {next_ae_step}: {ae_ids}"
                )
                start_time = time.time()

                continual_learning.train_autoencoder(
                    train_classes=ae_classes,
                    step_dir=ae_step_dir,
                    prev_ae_dir=prev_ae_dir,
                )

                continual_learning.create_ae_memory(
                    train_classes=ae_classes,
                    step_dir=ae_step_dir,
                    prev_ae_dir=prev_ae_dir,
                )

                elapsed = time.time() - start_time
                print(
                    f"AE sub-step {next_ae_step} complete in "
                    f"{elapsed // 60:.0f}m {elapsed % 60:.0f}s"
                )
                next_ae_step += 1

            embd_classes = get_train_classes(train_ids)
            embd_step_dir = get_step_dir(embd_run_dir, embd_step)
            prev_embd_dir = get_previous_step_dir(
                embd_run_dir, embd_step
            )
            write_step_train_ids(embd_step_dir, train_ids)

            # Use the exact AE checkpoint specified by the dependency list.
            prev_ae_dir = (
                None
                if ae_required_step is None
                else get_step_dir(ae_run_dir, ae_required_step)
            )

            print(f"Training embedder sub-step {embd_step}")
            print(f"Using AE memory from: {prev_ae_dir}")
            start_time = time.time()

            continual_learning.train_embedder(
                train_classes=embd_classes,
                step_dir=embd_step_dir,
                prev_embd_dir=prev_embd_dir,
                prev_ae_dir=prev_ae_dir,
            )

            continual_learning.create_embd_memory(
                train_classes=embd_classes,
                step_dir=embd_step_dir,
                prev_embd_dir=prev_embd_dir,
            )

            elapsed = time.time() - start_time
            print(
                f"Embedder sub-step {embd_step} complete in "
                f"{elapsed // 60:.0f}m {elapsed % 60:.0f}s"
            )

            if embd_step in run_test_indices:
                print(f"Testing embedder sub-step {embd_step}")
                continual_learning.test_embedder(
                    step_dir=embd_step_dir
                )


def run_batch(base_path, experiment_data):
    """Train the Embedder using the explicit batch schedule."""

    for run_position, run_data in enumerate(experiment_data):
        run_index = run_data.get("run_index", run_position)

        required_keys = {"batch_list", "batch_test_indices"}
        missing = required_keys - run_data.keys()

        if missing:
            raise ValueError(
                f"Run {run_index} is missing fields: "
                f"{sorted(missing)}. Regenerate the protocol data files."
            )

        _run_embedder_schedule(
            base_path=base_path,
            run_index=run_index,
            schedule=run_data["batch_list"],
            test_indices=run_data["batch_test_indices"],
            schedule_name="batch",
        )


def run_naive_cl(base_path, experiment_data):
    """Train the Embedder using the CL schedule, without AE replay."""

    for run_position, run_data in enumerate(experiment_data):
        run_index = run_data.get("run_index", run_position)

        required_keys = {"embd_list", "cl_test_indices"}
        missing = required_keys - run_data.keys()

        if missing:
            raise ValueError(
                f"Run {run_index} is missing fields: "
                f"{sorted(missing)}. Regenerate the protocol data files."
            )

        _run_embedder_schedule(
            base_path=base_path,
            run_index=run_index,
            schedule=run_data["embd_list"],
            test_indices=run_data["cl_test_indices"],
            schedule_name="naive_CL",
        )


def _run_embedder_schedule(
    base_path,
    run_index,
    schedule,
    test_indices,
    schedule_name,
):
    """Execute an Embedder-only schedule, without AE replay."""

    if any(
        not isinstance(index, int) or not 0 <= index < len(schedule)
        for index in test_indices
    ):
        raise ValueError(
            f"Run {run_index}: invalid test indices {test_indices!r} "
            f"for a schedule with {len(schedule)} steps."
        )

    embd_run_dir = os.path.join(
        base_path, "embedder", f"run_{run_index}"
    )
    os.makedirs(embd_run_dir, exist_ok=True)

    for step, train_ids in enumerate(schedule):
        train_classes = get_train_classes(train_ids)
        step_dir = get_step_dir(embd_run_dir, step)
        prev_embd_dir = get_previous_step_dir(embd_run_dir, step)

        write_step_train_ids(step_dir, train_ids)

        print(
            f"\nTraining {schedule_name} step {step}: {train_ids}"
        )

        continual_learning.train_embedder(
            train_classes=train_classes,
            step_dir=step_dir,
            prev_embd_dir=prev_embd_dir,
            prev_ae_dir=None,
        )

        continual_learning.create_embd_memory(
            train_classes=train_classes,
            step_dir=step_dir,
            prev_embd_dir=prev_embd_dir,
        )

        if step in test_indices:
            print(f"Testing {schedule_name} step {step}")
            continual_learning.test_embedder(step_dir=step_dir)


# ------------------------------------------------------------
# Reduced example
# ------------------------------------------------------------

def run_reduced_example(base_path):
    """Run the first two Protocol 1 embedder steps as a small demonstration."""
    protocol = "protocol_1"
    config = PROTOCOL_CONFIG[protocol]
    experiment_data = load_experiment_data(config["data_file"])

    example_path = os.path.join(
        base_path, "example", protocol, "CL_w_replay"
    )
    print("Running reduced Protocol 1 CL-with-replay example.")

    run_cl_w_replay(
        base_path=example_path,
        experiment_data=experiment_data[:1],
        max_runs=1,
        max_embedder_steps=2,
    )

    print(f"Reduced example completed. Outputs: {example_path}")


# ------------------------------------------------------------
# Command-line entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run continual face-recognition experiments."
    )
    parser.add_argument(
        "--base-path",
        default="outputs",
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        choices=sorted(PROTOCOL_CONFIG),
        default=list(PROTOCOL_CONFIG),
        help="Protocols to run (default: both).",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run a reduced Protocol 1 CL-with-replay example.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Regenerate both protocol experiment-data files before running.",
    )
    args = parser.parse_args()

    if args.generate_data:
        utils.generate_protocol_1_data()
        utils.generate_protocol_2_data()

    if args.example:
        run_reduced_example(args.base_path)
    else:
        main(args.base_path, args.protocols)
