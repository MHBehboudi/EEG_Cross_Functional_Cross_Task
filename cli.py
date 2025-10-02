import argparse
from .runner import train_and_build_zip

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mini", action="store_true", help="Use the mini release")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--outdir", type=str, default="output")
    args = p.parse_args()

    train_and_build_zip(outdir=args.outdir, mini=args.mini,
                        epochs=args.epochs, batch_size=args.batch_size,
                        num_workers=args.num_workers)

if __name__ == "__main__":
    main()
