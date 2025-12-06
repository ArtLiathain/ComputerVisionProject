#set page(width: 210mm, height: 297mm, margin: 25mm)
#set text(font: "Linux Libertine", size: 12pt)
#set par(justify: true)
#set heading(numbering: "1.")
#import "@preview/pintorita:0.1.4"
#show raw.where(lang: "pintora"): it => pintorita.render(it.text)

#align(center)[
  = Mini Project: Diffusion Models in Computer Vision  

Cascaded Shallow model \
Desirèe Charles \
Art Ó Liathain \
December 2025
]

#pagebreak()

= Abstract 
#v(1em)
Our project looks into the use of a cascaded shallow diffusion image generation pipeline for the CIFAR-10 dataset, which aims to distribute the three steps of coarse content generation, upsampling and refinement into three lightweight stages. The first one is a shallow conditional diffusion model working at a 16x16 resolution. The second step is a trained pixelShuffle upsampler that converts 16x16 images to 32x32. The last stage is a Vision Transformer (ViT) based refiner which is particularly thoughtful for 32x32 images since it uses global, attention based cleaning. A very modular structure is employed for the system so that every single stage can be replaced by a configuration object allowing automatic ablation across diffusion hyperparameters, model widths and depths, and teacher forcing schedules. The training process is gradual: the diffusion model is first trained, then the upsampler and refiner are trained together, and finally, all three stages are finetuned together. The performance is assessed by Kernel Inception Distance (KID), per stage mean squared error, and a perceptual LPIPS loss on the final stage, and also by qualitative visual inspection.

#pagebreak()

= Table of Contents
#outline()

#outline(
  title: [List of Figures],
  target: figure.where(kind: image)
)

#pagebreak()

= Introduction
#v(1em)

We focus on unconditional image generation on the CIFAR-10 dataset and implement a three stage “cascaded shallow diffusion” pipeline. The model begins with a diffusion model that creates a 16x16 image, and then the compact output is upsampled to 32x32 with a pixelshuffle that is trained, and finally, the upsampled image is improved through a Vision Transformer refiner. Each stage is intentionally shallow, and a configuration cell controls which concrete model is used at each stage (for example, diffusion versus a simple generator for the coarse stage, or a convolutional refiner versus a ViT refiner for the final stage). This makes it easy to switch architectures and run ablation studies by changing configuration values rather than rewriting the training code.

Goals:
- To implement and train a working diffusion based generator on CIFAR-10
- To design a cascaded architecture where each stage can be independently ablated
- To compare results across model size, training schedule and sample quality, using Kernel Inception Distance (KID) and qualitative visual inspection

#pagebreak()

= Background and Literature Review

== Diffusion models

*Need to find a paper for this section*

#pagebreak()
== Cascaded and multi-stage architectures


*Need to find a paper for this section.*


#pagebreak()

== Vision Transformers as refiners


*Need to find a paper for this section*

#pagebreak()

= Method

== Dataset and task
#v(1em)
Our project uses the CIFAR-10 dataset, which consists of 50,000 training and 10,000 test images (RGB) of size 32x32 across 10 object categories. In this project, the labels are ignored, and the task is image generation. Images are loaded and converted to tensors in (0,1). No significant augmentation is applied, as the quality of generative images is the primary concern rather than classification performance.
#v(1em)
== Cascaded shallow diffusion pipeline
#v(1em)
The generator is implemented as a three stage pipeline, which changes a high resolution tensor to:

- The diffusion stage: 16x16 coarse image
- The upsampler stage: 32x32 upsampled image
- The refiner stage: 32x32 refined image

A small "BaseStage" subclass (DiffusionStage, UpsamplerStage, RefinerStage) wraps each stage and the subclass includes the ways of preprocessing inputs, what the stage returns during the forward pass, and the ways of deriving training targets from the original image. This abstraction makes it possible to easily switch the different underlying models (like changing from a simple convolutional refiner to ViT refiner).
#v(1em)
=== Stage 1: Diffusion
#v(1em)
The generator for the coarse images is actually the "DiffusionBlock", which is encapsulated by the "DiffusionStage". The stage takes the input image of size 32x32 and first reduces its size to 16x16 by using bilinear interpolation. A random timestep is selected, after which Gaussian noise is added using the pre computed schedules. The stage keeps in its memory both the sample with noise and the actual noise. Inside the diffusion block is an initial 3x3 convolution that projects the input into a fixed channel width, then residual UNet blocks which are conditioned on a sinusoidal timestep embedding are applied. The network then gives a noise prediction of the same shape. The stage loss is computed as the mse between the predicted noise and the ground truth noise.
#v(1em)
=== Stage 2: Upsampler
#v(1em)
The second stage is a PixelShuffle Upsampler which is a learned upsampler and is also enclosed in UpsamplerStage. The upsampler is responsible for transforming an RGB image of 16x16 pixels to an RGB image of 32x32 pixels. The structure of the network consists of the following components: 
- A head 3x3 convolution that filters the input from 3 channels to a hidden width
- A PixelShuffle upsampling block that  doubles the spatial resolution
- A final 3x3 convolution that conducts the reverse process and translates back to 3 channels

The objective of the upsampler during the training session is the original ground truth picture that has been downsized to the same resolution as the output. The teacher forcing schedule determines whether the input to this stage is the ground truth image downsampled to 16x16 (teacher forcing) or the 16x16 output generated by the diffusion stage.
#v(1em)
=== Stage 3: Refiner

The last stage employs a Vision Transformer refiner to process the 32x32 image generated in the second stage. The architecture is based on the conventional ViT structure, which divides the input image into non-overlapping patches, uses a Conv2d patch embedding to project each patch to an embedding vector, and adds a learnable class token at the beginning. The positional embeddings are added, and the sequence of tokens goes through a series of transformer encoder blocks (Transformer block) comprising multihead self attention and feed forward MLPs. The refiner, does not take the class token as output. The class token is thrown away after the transformer stack, and only the patch tokens are kept. These tokens are then reshaped back into a 2D feature map and transposed convolution decoded to a 3 channels image using a convolution whose kernel size and stride correspond to the patch size.

== Training
#v(1em)
The training has three stages:

- *Stage 1: coarse diffusion *  

Only coarse diffusion. The isolation of the diffusion stage for a specific number of epochs is being the main characteristic of this phase. The coarse model's parameters are set to be trained while keeping the upsampler and refiner stages in their frozen state. The predicting noise and true noise at randomly sampled timesteps being MSE include the loss for this stage. 
#v(1em)

- *Stage 2: upsampler and refiner.*  

The coarse diffusion stage is put on hold and only the upsampler and refiner are trained. A teacher forcing schedule step by step lowers the possibility of giving the ground truth downsampled image to the upsampler and at the same time, increases the use of the coarse stage output that is made up of the raw images. The losses in this phase include: an MSE loss for the upsampler output compared to a resized ground truth image, a refiner loss that is comprised of MSE and a LPIPS.
#v(1em)

- *Stage 3: Fine tunning.*  
During the last stage, all three stages get unfreezed and get trained together with the same loss combinations. To stabilize learning, a separate teacher forcing schedule is applied.

Each phase is assigned its optimizer and learning rate scheduler, the settings are made through a configuration. The Config class is responsible for defining the core hyperparameters, such as diffusion schedule, the number of layers in the diffusion block, the dimensions and heights of the upsampler and refiner, and the teacher forcing start and end probabilities.
#v(1em)

== Evaluation metrics
#v(1em)
*Talk about: Kernel Inception Distance (KID), LPIPS loss, Time per epochs, FID score, Config used to create*

#pagebreak()

= Experimental analysis and evaluation

== Baseline configuration

*talk about baseline config*

#pagebreak()

== Ablation studies

* ablation studies*

== Results

*Add tables*

#pagebreak()

= Conclusion

*conclusion here*

#pagebreak()

= References

1. PixelShuffle: https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html 

