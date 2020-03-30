import argparse


def config(verbose=True):
    """Function to set parameters via command line arguments"""
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset', type=str, default="SS_S1")
    parser.add_argument('--splits_json', type=str, default="lila/SS_S1/SnapshotSerengetiSplits_v0.json")
    parser.add_argument('--dataset_json', type=str, default="lila/SS_S1/SnapshotSerengetiBboxes_20190903.json")
    #parser.add_argument('--dataset_json', type=str, default="lila/SS_S1/SnapshotSerengetiS01.json")
    #parser.add_argument('--bboxes_json', type=str, default="lila/SS_S1/SnapshotSerengetiBboxes_20190903.json")
    parser.add_argument('--data_dir', type=str, default="../../../scratch/zwinzurk/wild/datasets")
    parser.add_argument('--wrangler', type=bool, default=True)  # run wrangler script
    parser.add_argument('--presave', type=bool, default=False)  # presave tensors
    # task
    parser.add_argument('--task', type=str, default="classification")  # classification or detection
    parser.add_argument('--level', type=str, default="species")  # species or animal
    parser.add_argument('--arch', type=str, default="resnet50")
    parser.add_argument('--runoffset', type=int, default=0)  # number of prior runs
    parser.add_argument('--runs', type=int, default=1)  # number of runs
    parser.add_argument('--load_weights', type=str, default=None)  # path to state dict if finetuning
    # hyperparameters
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64*8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--accumulation_steps', type=int, default=1)  # Optional: steps for gradient accumulation
    parser.add_argument('--resize_dim', type=tuple, default=(300, 500))

    args = parser.parse_args()

    if verbose:
        print(f'Dataset: {args.dataset}')
        print(f'Data Dir: {args.data_dir}')
        print(f'Batch Size: {args.batch_size}')
        print(f'Architecture: {args.arch}')
        print(f'Weights: {args.load_weights}')
        print(f'Task: {args.task}')
        print(f'Level: {args.level}')
        print(f'Runs: {args.runs}\n')

    return args
