import argparse
import os
import subprocess

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="unlearning algorithm")
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--score", default="accuracy", help="{accuracy, ulira, umia}")

    args, unknownargs = parser.parse_known_args()

    for forget_size in (10,):
        os.makedirs(f"results/{args.score}/{args.method}/forget_size={forget_size}", exist_ok=True)
        os.system(f"rm results/{args.score}/{args.method}/forget_size={forget_size}/{args.net}.txt")

        if args.score == "accuracy":
            for i in range(128):
                out = subprocess.check_output(
                    [
                        "python",
                        "score.py",
                        "--method",
                        args.method,
                        "--split_idx",
                        str(i),
                        "--net",
                        args.net,
                        "--forget_size",
                        str(forget_size),
                    ]
                )
                with open(f"results/{args.score}/{args.method}/forget_size={forget_size}/{args.net}.txt", "a") as file:
                    print(out[1:-2], file=file)

        if args.score in ("ulira", "umia"):
            out = subprocess.check_output(
                [
                    "python",
                    "attack.py",
                    "--method",
                    args.method,
                    "--net",
                    args.net,
                    "--forget_size",
                    str(forget_size),
                    "--attack",
                    args.score,
                    "--plot_distr",
                ]
            )
            with open(f"results/{args.score}/{args.method}/forget_size={forget_size}/{args.net}.txt", "w") as file:
                print(out, file=file)
