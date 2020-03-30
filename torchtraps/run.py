from config import config
from magic import wrangler
from models import get_model
from train import train_model
# from inference import inference

if __name__ == "__main__":

    args = config(verbose=True)
    if args.wrangler:
        wrangler(args)
    for i in range(args.runoffset+1, args.runoffset+args.runs+1):
        model = get_model(args)
        model = train_model(model, args, i)
