import os.path

from Baby import *
from utils import *
from config import *
from dataset import *

train_data_path = os.path.join(Config.data_path, 'train')
test_data_path = os.path.join(Config.data_path, 'test')

dataset_link_path = Config.dataset_link_path

# helper functions:

def print_num_vectors_stats(num_vectors: dict):

    values = list(num_vectors.values())

    if not values:
        print("num_vectors is empty")
        return

    median_val = np.median(values)
    max_val = max(values)
    min_val = min(values)

    print(f"num_vectors: median {median_val}, max {max_val}, min {min_val}")


def prepare_train_dataset(train_classes, extra_data_paths=None, extra_classes=None):

    if extra_data_paths is None:
        data_paths = [train_data_path]
        classes = [train_classes]
    else:
        data_paths = [train_data_path, extra_data_paths]
        classes = [train_classes, extra_classes]

    dataset_link_path_train = prepare_dataset_links(
        os.path.join(dataset_link_path, 'train'),
        data_paths,
        classes
    )

    return dataset_link_path_train


def create_trainloader(opt):

    train_dataset = Dataset(
        opt.train_root,
        opt.train_list,
        phase='train',
        input_shape=opt.input_shape
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )

    return trainloader


def train_embedder_model(cl_baby, trainloader, opt, get_from_checkpoint=False,
                         checkpoint_name=''):

    cl_baby.train(
        dataloaders=trainloader,
        train_method='arcface_resnet',
        model_type='',
        num_epochs=0,
        embd_len=0,
        pretrained=False,
        prev_classes=[],
        current_classes=[],
        get_from_checkpoint=get_from_checkpoint,
        checkpoint_name=checkpoint_name,
        opt=opt
    )


def print_ae_memory_stats(ae_memory):

    print('identities in loaded AE memory:')

    for key, val in ae_memory.memory.items():
        print(f'class in memory: {key}, num clusters: {len(val["centroids"])}')
        print_num_vectors_stats(val['num_vectors'])


# experimental functions:

def train_autoencoder(train_classes, step_dir, prev_ae_dir=None):

    print(
        f'train autoencoder, step_dir: {step_dir}, '
        f'prev_ae_dir: {prev_ae_dir}, train_classes: {train_classes}'
    )

    cl_baby = Baby.Baby()
    opt = Config()

    dataset_link_path_train = prepare_train_dataset(train_classes)

    opt.train_root = dataset_link_path_train
    opt.checkpoints_path = step_dir

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    os.makedirs(
        os.path.join(opt.checkpoints_path, 'sample_images'),
        exist_ok=True
    )

    write_image_names(opt)

    if prev_ae_dir is None:
        opt.resume = ''
    else:
        opt.resume = prev_ae_dir

    cl_baby.train_ae2(
        dataset_link_path_train,
        opt.checkpoints_path,
        read_from_disk=True,
        resume_dir=opt.resume,
        get_from_checkpoint=False
    )


def create_ae_memory(train_classes, step_dir, prev_ae_dir=None):

    print(
        f'create_ae_memory, step_dir: {step_dir}, '
        f'prev_ae_dir: {prev_ae_dir}, train_classes: {train_classes}'
    )

    cl_baby = Baby.Baby()
    opt = Config()

    dataset_link_path_train = prepare_train_dataset(train_classes)

    opt.train_root = dataset_link_path_train
    opt.checkpoints_path = step_dir

    write_image_names(opt)

    # Load the current AE model.
    cl_baby.train_ae2(
        dataset_link_path_train,
        opt.checkpoints_path,
        read_from_disk=True,
        resume_dir=opt.checkpoints_path,
        get_from_checkpoint=True
    )

    # Load previous AE memory, if there is one.
    if prev_ae_dir is not None:

        print(f'loading previous ae memory from {prev_ae_dir}')

        cl_baby.load_from_disk(prev_ae_dir)

        print_ae_memory_stats(cl_baby.ae_memory)

    # Update memory with the current real identities.
    cl_baby.build_ae_memory(
        dataset_link_path_train,
        clear_prev_memory=False,
        clear_prev_ids=True
    )

    cl_baby.save(step_dir)


def train_embedder(train_classes, step_dir, prev_embd_dir=None, prev_ae_dir=None):

    print(
        f'train_embedder, step_dir: {step_dir}, '
        f'prev_embd_dir: {prev_embd_dir}, '
        f'prev_ae_dir: {prev_ae_dir}, '
        f'train_classes: {train_classes}'
    )

    cl_baby = Baby.Baby()
    opt = Config()

    opt.checkpoints_path = step_dir

    os.makedirs(opt.checkpoints_path, exist_ok=True)

    real_classes = train_classes

    print('real classes:')
    print(real_classes)

    gen_classes = []

    # If a previous AE exists, use its memory for replay.
    if prev_ae_dir is not None:

        print(f'loading AE memory from {prev_ae_dir}')

        cl_baby.load_from_disk(prev_ae_dir)

        ae_memory = cl_baby.ae_memory

        print('identities in loaded AE memory:')

        for key, _ in ae_memory.memory.items():
            print(f'class: {key}')

            if key not in real_classes:
                gen_classes.append(key)

        print('generated classes:')
        print(gen_classes)

    # Prepare training data.
    if len(gen_classes) > 0:

        gen_root = os.path.join(step_dir, 'generated_images')

        os.makedirs(gen_root, exist_ok=True)

        # Load the AE from the previous AE directory.
        cl_baby.train_ae2(
            '',
            '',
            read_from_disk=True,
            resume_dir=prev_ae_dir,
            get_from_checkpoint=True
        )

        cl_baby.generate_images_from_ae2(
            gen_root,
            gen_classes
        )

        print("finished generate_images_from_ae2")

        dataset_link_path_train = prepare_train_dataset(
            real_classes,
            extra_data_paths=gen_root,
            extra_classes=gen_classes
        )

    else:

        dataset_link_path_train = prepare_train_dataset(real_classes)

    opt.train_root = dataset_link_path_train

    write_image_names(opt)

    trainloader = create_trainloader(opt)

    # Resume the previous embedder, if one exists.
    if prev_embd_dir is None:
        opt.resume = ''
    else:
        opt.resume = os.path.join(
            prev_embd_dir,
            'resnet18_best.pth'
        )

    train_embedder_model(
        cl_baby,
        trainloader,
        opt,
        get_from_checkpoint=False,
        checkpoint_name=''
    )


def create_embd_memory(train_classes, step_dir, prev_embd_dir=None):

    print(
        f'create_embd_memory, step_dir: {step_dir}, '
        f'prev_embd_dir: {prev_embd_dir}, '
        f'train_classes: {train_classes}'
    )

    cl_baby = Baby.Baby()
    opt = Config()

    dataset_link_path_train = prepare_train_dataset(train_classes)

    opt.train_root = dataset_link_path_train
    opt.checkpoints_path = step_dir

    write_image_names(opt)

    trainloader = create_trainloader(opt)

    # Load the current embedder.
    train_embedder_model(
        cl_baby,
        trainloader,
        opt,
        get_from_checkpoint=True,
        checkpoint_name='resnet18_best.pth'
    )

    # Load previous embedding memory.
    if prev_embd_dir is not None:

        print(f'loading previous embedding memory from {prev_embd_dir}')

        cl_baby.load_from_disk(prev_embd_dir)

        print('identities in loaded embedding memory:')

        embd_memory = cl_baby.embd_memory

        for key, _ in embd_memory.memory.items():
            print(f'class_name: {key}')

    # Build/update memory.
    cl_baby.build_embd_memory(
        trainloader,
        clear_prev_memory=False,
        clear_prev_ids=True
    )

    cl_baby.save(opt.checkpoints_path)


def test_embedder(step_dir):

    print(f'test_embedder, step_dir: {step_dir}')

    cl_baby = Baby.Baby()
    opt = Config()

    opt.checkpoints_path = step_dir

    # Load current embedder.
    train_embedder_model(
        cl_baby,
        [],
        opt,
        get_from_checkpoint=True,
        checkpoint_name=''
    )

    # Load embedding memory.
    cl_baby.load_from_disk(opt.checkpoints_path)

    test_classes = []

    embd_memory = cl_baby.embd_memory

    print('identities in embedding memory:')

    for key, val in embd_memory.memory.items():
        test_classes.append(key)

        print(
            f'class in memory: {key}, '
            f'num clusters: {len(val["centroids"])}'
        )
        print_num_vectors_stats(val['num_vectors'])

    # Prepare test dataset.
    test_classes_nested = [test_classes]

    dataset_link_path_test = prepare_dataset_links(
        os.path.join(dataset_link_path, 'test'),
        [test_data_path],
        test_classes_nested
    )

    opt.train_root = dataset_link_path_test

    write_image_names(opt)

    test_dataset = Dataset(
        opt.train_root,
        opt.train_list,
        phase='test',
        input_shape=opt.input_shape
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.train_batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )

    performance_df = pd.DataFrame(
        columns=['cls_name', 'acc']
    )

    print('test classes:')
    print(test_classes)

    cl_baby.test(
        testloader,
        '',
        test_classes,
        performance_df
    )

    lfw_acc = arcface_resnet_test(
        opt,
        cl_baby.embd_model
    )

    performance_df.loc[len(performance_df.index)] = [
        'lfw_acc',
        lfw_acc
    ]

    performance_df.to_csv(
        os.path.join(
            opt.checkpoints_path,
            'performance_all_faces.csv'
        )
    )


def test_embedder_verification(step_dir):

    print(f'test_embedder_verification, step_dir: {step_dir}')

    cl_baby = Baby.Baby()
    opt = Config()

    opt.checkpoints_path = step_dir

    # Load current embedder.
    train_embedder_model(
        cl_baby,
        [],
        opt,
        get_from_checkpoint=True,
        checkpoint_name=''
    )

    # Load embedding memory.
    cl_baby.load_from_disk(opt.checkpoints_path)

    test_classes = []

    embd_memory = cl_baby.embd_memory

    print('identities in embedding memory:')

    for key, val in embd_memory.memory.items():
        test_classes.append(key)

        print(
            f'class in memory: {key}, '
            f'num clusters: {len(val["centroids"])}'
        )
        print_num_vectors_stats(val['num_vectors'])

    # Prepare test dataset.
    test_classes_nested = [test_classes]

    dataset_link_path_test = prepare_dataset_links(
        os.path.join(dataset_link_path, 'test'),
        [test_data_path],
        test_classes_nested
    )

    opt.train_root = dataset_link_path_test

    write_image_names(opt)

    # Generate familiar/unfamiliar verification pairs.
    verification_pairs_file = os.path.join(
        opt.checkpoints_path,
        'verification_pairs_all.txt'
    )

    generate_face_verification_pairs_all(
        test_data_path,
        test_classes,
        verification_pairs_file
    )

    # Test verification.
    cl_baby.test_verification(
        verification_pairs_file
    )

