import argparse
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_split_idx", default=0, type=int)
    parser.add_argument("--stop_split_idx", default=128, type=int)
    parser.add_argument("--method", type=str, help="unlearning algorithm")

    args, unknownargs = parser.parse_known_args()

    for i in range(args.start_split_idx, args.stop_split_idx):
        print(f"\nTrain model {i}")

        subprocess.call(["python", f"unlearning/{args.method}.py", "--split_idx", str(i)] + unknownargs)
