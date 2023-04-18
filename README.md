# huggingNanoGPT

🤗 Transformers style model that's compatible with [nanoGPT](https://github.com/karpathy/nanoGPT) checkpoints.

The 🤗 ecosystem is expansive, but not particularly optimized for pre-training. nanoGPT is a great low-overhead way to get into pre-training, but it has a limited ecosystem, and lacks some creature comforts.

The `NanoGPTLMHeadModel` implementation in this repository is very similar to `GPT2LMHeadModel`, but it uses nanoGPT conventions for configuration, uses `nn.Linear` in place of `nn.Conv1D` in a few places, and adds the ability to enable/disable bias parameters like nanoGPT does.

Most likely, you would use this by pretraining and/or finetuning using nanoGPT, and then using 🤗 Transformers for other kinds of training like RLHF or for its nicer inference pipeline. 

## Why not just use `GPT2LMHeadModel`?

You're right, it's not that difficult to jam the weights from `nanoGPT` into `GPT2LMHeadModel`.

However, the models have some minor differences. I suspect that these differences are not a big deal for inference-only use cases, but for training, I'd rather have zero air-gaps. This model is as close as it gets and should behave like a totally native model in the 🤗 Transformers world. 


## TODO / Improvements

- Test to make sure that this works for more than just inference
- Implement Flash Attention to improve performance
- Clean up the code, remove some optional non-nanoGPT supported features, etc. 
- Support saving nanoGPT compatible checkpoints

## Usage

    from hugging_nanogpt import NanoGPTLMHeadModel
    
    hf_model = NanoGPTLMHeadModel.from_nanogpt_ckpt('/path/to/ckpt.pt').cuda()
    hf_model.generate(...)

## License

This code incorporates portions of [🤗 Transformers](https://github.com/huggingface/transformers) and [nanoGPT](https://github.com/karpathy/nanoGPT). It is released under the Apache 2.0 License
