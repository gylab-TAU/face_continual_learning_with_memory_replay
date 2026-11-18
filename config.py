class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 2000 # 1000 # 500 # 200 # 13938
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'


    display = False
    finetune = False

    resume = ''
    train_root = ''
    train_list = 'train_list.txt'
    val_list = train_list

    # Dataset paths
    # Set this to the absolute path of the image gallery.
    data_path = '/path/to/cl_data/gallery/'
    # Set this to the absolute path of links dir.
    dataset_link_path = '/path/to/cl_data/gallery/'

    all_identities = [
        "addison", "alex", "april", "arizona", "callie",
        "cristina", "derek", "dwight", "george", "izzie",
        "jackson", "lexie", "mark", "meredith", "michael",
        "miranda", "owen", "pam", "richard", "teddy",
    ]

    test_root = ''
    test_list = 'test.txt'

    # put absolute paths here:
    lfw_root = '/path/to/lfw-align-128'
    lfw_test_list = '/path/to/lfw_test_pair.txt'

    checkpoints_path = ''
    load_model_path = ''
    test_model_path = ''

    save_interval = 10

    train_batch_size = 96  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 10 # 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 100
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
