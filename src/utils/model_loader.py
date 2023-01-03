from models import NetGAT, NetGCN

def get_model(args, device="cpu", num_features=None, num_classes=None):

    if args.model == "GAT":
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGAT(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
        )
        return model
    elif args.model == "GCN":
        emb_sizes = [args.emb_dim] * (args.num_layers + 1)
        model = NetGCN(
            num_features,
            num_classes,
            emb_sizes=emb_sizes,
            device=device,
            scatter=args.scatter,
            drpt_prob=args.dropout,
        )
        return model


