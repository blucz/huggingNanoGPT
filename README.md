# huggingNanoGPT

🤗 Transformers style model that's compatible with [nanoGPT](https://github.com/karpathy/nanoGPT) checkpoints.

The 🤗 ecosystem is expansive, but not particularly optimized for pre-training. nanoGPT is a great low-overhead way to get into pre-training, but it has a limited ecosystem, and lacks some creature comforts.

The `NanoGPTLMHeadModel` implementation in this repository is very similar to `GPT2LMHeadModel`, but it uses nanoGPT conventions for configuration, uses `nn.Linear` in place of `nn.Conv1D` in a few places, and adds the ability to enable/disable bias parameters. 

Most likely, you would use this by pretraining or finetuning using nanoGPT first, and then using 🤗 Transformers for other kinds of training like RLHF or inference.

## Why not just use `GPT2LMHeadModel`?

So, you can do this, and it's not that hard to jam the weights from `nanoGPT` into there.

However, the models have some minor differences. I suspect that these differences are not a big deal for inference-only use cases, but for training, I'd rather things be precise. This model gets things as identical as possible. 

According to [https://huggingface.co/blog/transformers-design-philosophy](The 🤗 people), the right thing to do in a situation like this is to fork the model, and that's what this repository is all about. nanoGPT isn't quite GPT2, it's better than GPT2 in some minor ways, and it deserves to be its own thing. 

## TODO / Improvements

- Test to make sure that this works for more than just inference
- Implement Flash Attention to improve performance
- Clean up the code, remove some optional non-nanoGPT supported features, etc. 
- Support converting back to a nanoGPT compatible checkpoint

## Usage

    from hugging_nano_gpt import NanoGPTLMHeadModel
    
    hf_model = NanoGPTLMHeadModel.from_nanogpt_ckpt('/path/to/ckpt.pt').cuda()
    hf_model.generate(...)

## License

This code incorporates portions of [🤗 Transformers](https://github.com/huggingface/transformers) and [nanoGPT](https://github.com/karpathy/nanoGPT). It is released under the Apache 2.0 License
