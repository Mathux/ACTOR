from src.models.get_model import LOSSES, MODELTYPES, ARCHINAMES, JOINTSTYPES


def add_model_options(parser):
    group = parser.add_argument_group('Model options')
    group.add_argument("--modelname", help="Choice of the model, should be like cvae_transformer_rc_rcxyz_kl")
    group.add_argument("--latent_dim", default=256, type=int, help="dimensionality of the latent space")
    group.add_argument("--lambda_kl", required=True, type=float, help="weight of the kl divergence loss")
    group.add_argument("--lambda_rc", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--lambda_rcxyz", default=1.0, type=float, help="weight of the rc divergence loss")
    group.add_argument("--jointstype", default="vertices", choices=JOINTSTYPES, help="Jointstype for training with xyz")

    group.add_argument('--vertstrans', dest='vertstrans', action='store_true', help="Training with vertex translations in the SMPL mesh")
    group.add_argument('--no-vertstrans', dest='vertstrans', action='store_false', help="Training without vertex translations in the SMPL mesh")
    group.set_defaults(vertstrans=False)

    group.add_argument("--num_layers", default=4, type=int, help="Number of layers for GRU and transformer")
    group.add_argument("--activation", default="gelu", help="Activation for function for the transformer layers")

    # Ablations
    group.add_argument("--ablation", choices=[None, "average_encoder", "zandtime", "time_encoding", "concat_bias"],
                       help="Ablations for the transformer architechture")


def parse_modelname(modelname):
    modeltype, archiname, *losses = modelname.split("_")

    if modeltype not in MODELTYPES:
        raise NotImplementedError("This type of model is not implemented.")
    if archiname not in ARCHINAMES:
        raise NotImplementedError("This architechture is not implemented.")

    if len(losses) == 0:
        raise NotImplementedError("You have to specify at least one loss function.")

    for loss in losses:
        if loss not in LOSSES:
            raise NotImplementedError("This loss is not implemented.")

    return modeltype, archiname, losses
