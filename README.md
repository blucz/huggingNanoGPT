# huggingNanoGPT

🤗transformers style model that is compatible with [nanoGPT](https://github.com/karpathy/nanoGPT) checkpoints.

The 🤗 ecosystem is expansive, but not particularly optimized for training. nanoGPT is great for low-overhead training but has a limited ecosystem and limited flexibility for inference. For example, there is no real alternative to the [trl](https://github.com/lvwerra/trl) library for nanoGPT, and the inference pipeline included with nanoGPT is missing creature comforts like beam search and repetition penalty.

This 🤗transformers model attempts to blend the best of both worlds. Use nanoGPT for fast, manageable pretraining, then 🤗transformers for the nice ecosystem afterwards. 

## Why not just use `GPT2LMHeadModel`?

So, you can do this, and it's not that hard to jam the weights from `nanoGPT` into there.

However, the models have some minor differences. In particular, nanoGPT defaults to `bias=False` for most of the layers, whereas 🤗transformers makes bias parameters mandatory. Also, there are a few places where `Conv1D` layers are used in 🤗transformers and `Linear` layers are used in nanoGPT. This model trues up those differences to get things as identical as possible. 

Finally, `GPT2Config` doesn't exactly match the nanoGPT config structure, and it's nice to have them match 1-1 to prevent mistakes. 

According to [https://huggingface.co/blog/transformers-design-philosophy](The 🤗 people), the right thing to do in a situation like this is to fork the model, and that's what this repository is all about. 

## TODO / Improvements

- Test to make sure that this works for more than just inference
- Implement Flash Attention to improve performance
- Clean up the code, remove some optional non-nanoGPT supported features, etc. 
- Support converting back to a nanoGPT compatible checkpoint

## Usage

    from hugging_nano_gpt import NanoGPTLMHeadModel
    
    hf_model = NanoGPTLMHeadModel.from_nanogpt_ckpt('/path/to/ckpt.pt').cuda()
    hf_model.generate(...)

