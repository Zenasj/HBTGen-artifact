from huggingface import HuggingfaceRunner, main
args = ["--performance", "--inference", "--backend", "inductor", "--progress", "--device", "cpu"]
main(HuggingfaceRunner(), args=args)

subprocess.check_call(
                    [sys.executable] + sys.argv + [f"--only={name}"],
                    timeout=timeout,
                    env=env,
                )

if len(sys.argv) == 1:
    sys.argv.extend(args)
else:
    args = sys.argv[1:]