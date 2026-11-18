import cv2
import numpy as np
import torch
from sklearn.metrics import accuracy_score

def load_image(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    image = cv2.resize(image, (128, 128))

    # normalize
    image = image.astype(np.float32)
    image -= 127.5
    image /= 127.5

    # (H,W) -> (C,H,W)
    image = np.expand_dims(image, axis=0)

    # (C,H,W) -> (B,C,H,W)
    image = np.expand_dims(image, axis=0)

    return image

def get_embedding(model, img_path, device="cuda"):

    image = load_image(img_path)

    if image is None:
        return None

    data = torch.from_numpy(image).to(device)

    with torch.no_grad():
        output = model(data)

    embedding = output.cpu().numpy()

    # remove batch dimension only
    embedding = embedding[0]

    # normalize
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


def cosine_similarity(emb1, emb2):
    """
    Cosine similarity between two embeddings.
    """

    return np.dot(emb1, emb2)


def find_best_threshold(scores, labels):
    """
    Find threshold maximizing verification accuracy.

    Args:
        scores:
            cosine similarities

        labels:
            1 = same identity
            0 = different identity

    Returns:
        best threshold
        best accuracy
    """

    thresholds = np.linspace(
        min(scores),
        max(scores),
        1000
    )

    best_acc = 0
    best_threshold = 0

    for t in thresholds:

        predictions = (
            np.array(scores) >= t
        ).astype(int)

        acc = accuracy_score(
            labels,
            predictions
        )

        if acc > best_acc:
            best_acc = acc
            best_threshold = t

    return best_threshold, best_acc


def evaluate_subset(scores, labels, mask):
    """
    Evaluate a subset of pairs.
    """

    scores_sub = [
        s for s, m in zip(scores, mask)
        if m
    ]

    labels_sub = [
        y for y, m in zip(labels, mask)
        if m
    ]

    if len(scores_sub) == 0:
        return None, None

    return find_best_threshold(
        scores_sub,
        labels_sub
    )


def evaluate_verification(
        model,
        pair_file,
        device="cuda"
):
    """
    Run verification evaluation.

    Returns:
        dictionary with accuracies
    """

    model.eval()

    scores = []
    labels = []
    train_flags = []

    # cache embeddings
    embedding_cache = {}

    with open(pair_file, "r") as f:

        lines = f.readlines()

    for line in lines:

        img1, img2, same, train = line.strip().split(",")

        same = int(same)
        train = int(train)

        # -----------------------------
        # embedding cache
        # -----------------------------

        if img1 not in embedding_cache:

            embedding_cache[img1] = get_embedding(
                model,
                img1,
                device
            )

        if img2 not in embedding_cache:

            embedding_cache[img2] = get_embedding(
                model,
                img2,
                device
            )

        emb1 = embedding_cache[img1]
        emb2 = embedding_cache[img2]

        if emb1 is None or emb2 is None:
            continue


        sim = cosine_similarity(
            emb1,
            emb2
        )

        scores.append(sim)
        labels.append(same)
        train_flags.append(train)


    # ------------------------------------
    # Evaluate all pairs
    # ------------------------------------

    all_threshold, all_acc = find_best_threshold(
        scores,
        labels
    )


    # ------------------------------------
    # Train pairs
    # ------------------------------------

    train_mask = [
        x == 1 for x in train_flags
    ]

    train_threshold, train_acc = evaluate_subset(
        scores,
        labels,
        train_mask
    )


    # ------------------------------------
    # Non-train pairs
    # ------------------------------------

    non_train_mask = [
        x == 0 for x in train_flags
    ]

    non_train_threshold, non_train_acc = evaluate_subset(
        scores,
        labels,
        non_train_mask
    )


    results = {

        "num_pairs": len(scores),

        "all":
        {
            "threshold": all_threshold,
            "accuracy": all_acc
        },

        "train":
        {
            "num_pairs": sum(train_mask),
            "threshold": train_threshold,
            "accuracy": train_acc
        },

        "non_train":
        {
            "num_pairs": sum(non_train_mask),
            "threshold": non_train_threshold,
            "accuracy": non_train_acc
        }
    }


    return results