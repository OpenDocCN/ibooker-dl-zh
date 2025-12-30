# Chapter 17\. Speeding Up Transformers

In Chapters [15](ch15.html#transformer_chapter) and [16](ch16.html#vit_chapter), we built all kinds of transformers, from classifiers, translators and chatbots, to vision and multimodal transformers. While transformers are incredibly versatile and powerful, they are far from perfect. In particular, they can be very slow, especially when processing long input sequences.

Luckily, many techniques have been developed to speed up transformers of any size:

*   To speed up decoding in generative transformers, we will use key/value caching and speculative decoding, then we will take of a quick look at several approaches to parallelize text generation.

*   To accelerate multi-head attention (MHA), which is one of the most computationally expensive components of transformers, we will look at sparse attention, approximate attention, sharing projections, and FlashAttention.

*   To speed up gigantic transformers of up to trillions of parameters, we will discuss mixture of experts (MoE).

*   To train large transformers efficiently, we will discuss parameter-efficient fine-tuning (PEFT) using adapters such as Low-Rank Adaptation (LoRA), activation checkpointing, sequence packing, gradient accumulation, and parallelism.

###### Tip

Another way to speed up a transformer is to make it smaller. This can be done using reduced precision and quantization, which are discussed in [Appendix B](app02.html#precision_appendix).

That’s quite a lot of techniques to cover, and they are fairly advanced, so you can safely skip this chapter for now if you are new to transformers, and come back later whenever needed. This is why this chapter is online-only, available at [*https://homl.info*](https://homl.info), to leave room for the other chapters.