from __future__ import print_function


def model_opts(parser):
    group = parser.add_argument_group("NeRF")

    group.add(
        "--activation",
        "-activation",
        type=str,
        default="relu",
        choices=["relu", "relu6", "elu"],
        help="Activation function for hidden layers of neural net."
        "Options are:"
        "[relu|relu6|elu]",
    )

    group.add(
        "--hidden_dim",
        "-hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension of neural network.",
    )

    group = parser.add_argument_group("Hyperparameters")
    group.add(
        "--batch_size",
        "-batch_size",
        type=int,
        default=1024,
        help="Maximum batch size for training.",
    )

    group.add(
        "--lr",
        "-lr",
        type=float,
        default=1e-2,
        help="Learning rate for training the network.",
    )

    group.add(
        "--learnable_positional_encoding",
        "-learnable_positional_encoding",
        action="store_true",
        help="If passed, uses learnable positional encodings based on foruier features.",
    )


def data_opts(parser):
    group = parser.add_argument_group("General")

    group.add(
        "--num_workers",
        "-num_workers",
        type=int,
        default=12,
        help="Num workers for dataloader.",
    )

    group.add(
        "--inp_path",
        "-inp_path",
        type=str,
        required=True,
        help="System path to 2D image for training NeRF.",
    )

    group.add(
        "--save_dir",
        "-save_dir",
        type=str,
        required=True,
        help="System path to expt dir.",
    )

    group.add(
        "--log_dir",
        "-log_dir",
        type=str,
        required=True,
        help="System path to tensorboard logs.",
    )

    group.add(
        "--use_positional_encoding",
        "-use_positional_encoding",
        action="store_true",
        help="If passed, uses positional encodings based on foruier features.",
    )

    group.add(
        "--use_random_fourier_features",
        "-use_random_fourier_features",
        action="store_true",
        help="If passed, uses fixed random fourier features instead of positions.",
    )

    group.add(
        "--devices",
        "-devices",
        type=int,
        default=0,
        help="Number of GPUs available on the system.",
    )

    group.add(
        "--max_freq_exp",
        "-max_freq_exp",
        type=int,
        default=10,
        help="Range of angular frequencies to extract positional encodings.",
    )


def trainer_opts(parser):
    group = parser.add_argument_group("General")

    group.add(
        "--save_every",
        "-save_every",
        type=int,
        default=200,
        help="Interval (epochs) after which to save checkpoints.",
    )

    group.add(
        "--save_last",
        "-save_last",
        action="store_true",
        help="If passed, saves the ckpt for the last epoch.",
    )

    group.add(
        "--check_val_every_n_epoch",
        "-check_val_every_n_epoch",
        type=int,
        default=20,
        help="Interval (epochs) after which to validate model.",
    )

    group.add(
        "--epochs",
        "-epochs",
        type=int,
        default=1000,
        help="Number of iterations that the network should be trained for.",
    )
