import torch
torch.set_float32_matmul_precision("high")

from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
import torch.utils.benchmark as benchmark
import argparse
import gc

def benchmark_fn(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},
        num_threads=torch.get_num_threads(),
    )
    return f"{(t0.blocked_autorange().mean):.3f}"


def call_pipeline(pipeline, prompt, prior_output,  negative_prompt):
    _ = pipeline(
        image_embeddings=prior_output,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10
    )

def main(args):
    prompt = "A colorful video message that shows a wizard with floating text that says 'achievement unlocked!'"
    negative_prompt = "warped text, blurry text, missing letters, missing words"
    
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.bfloat16).to("cuda")
    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        guidance_scale=7.5,
        num_images_per_prompt=1,
        num_inference_steps=30
    )
    prior_output = prior_output.image_embeddings.to(torch.float16)
    
    del prior
    torch.cuda.empty_cache()
    gc.collect()

    decoder = StableCascadeDecoderPipeline.from_pretrained(
        "stabilityai/stable-cascade", torch_dtype=torch.float16
    ).to("cuda")

    if args.compile:
        # Compiler settings.
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        
        # Compiler.
        decoder.decoder.to(memory_format=torch.channels_last)
        decoder.decoder = torch.compile(decoder.decoder, mode="max-autotune", fullgraph=True)
        decoder.vqgan.to(memory_format=torch.channels_last)
        decoder.vqgan = torch.compile(decoder.vqgan, mode="max-autotune", fullgraph=True)
    
    decoder.set_progress_bar_config(disable=True)
    print(decoder.components.keys())

    # warm-up 
    for _ in range(3):
        call_pipeline(decoder, prompt, prior_output, negative_prompt)

    time = benchmark_fn(call_pipeline, decoder, prompt, prior_output, negative_prompt)

    if args.save_outputs:
        decoder_output = decoder(
            image_embeddings=prior_output,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=10
        ).images[0]
        decoder_output.save("cascade.png")

    return time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    args = parser.parse_args()

    time = main(args)
    print(f"Batch size: 1 in {time} seconds with compile: {args.compile}")