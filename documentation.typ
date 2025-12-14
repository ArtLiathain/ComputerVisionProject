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


= Table of Contents
#outline()

#outline(
  title: [List of Figures],
  target: figure.where(kind: image)
)

#pagebreak()

= Introduction
#v(1em)
Recent advances in generative modeling have shown that high-quality image synthesis can be achieved not only through large, monolithic architectures, but also through the composition of multiple simpler models.
Cascaded generative approaches decompose the image generation task into a sequence of stages, where each stage is responsible for adding structure, resolution, or detail.
By progressively refining an image rather than generating it in a single step, these models can reduce training complexity, improve interpretability, and allow individual components to be analyzed or modified independently.

Each stage serves a different purpose, the first is a generation model to generate a coarse 16x16 resolution image.
The second stage is a learned upsampler that upsamples the 16x16 coarse output images to 32x32.
The last stage is a refiner which refines the output of the learned upsampler using perception loss to an accurate 32x32 representation.
These models cascade from one to another to approximate a CIFAR-10 image while each retaining a shallow number of layers favouring wide by shallow models over the traditional deep and narrow of current state of the art.

The goals of this paper are:
- To implement and train a working cascading image generator on CIFAR-10
- To implement a wide variety of models to compare in the different stages
- To compare results across model size, training schedule and sample quality, using Kernel Inception Distance (KID) and qualitative visual inspection

#figure(
    image("images/Real.png", width: 80%),
  caption: "Generated samples."
)

#figure(
    image("images/Target.png", width: 80%),
  caption: "Target samples."
)


= Background and Literature Review
Within this project a wide variety of models have been selected due to their prominence in state of the art computer vision. This paper assumes a base understanding of computer vision models as the focus of the paper is cascading image generation.
== Generation models 

=== Denoising Autoencoders (DAE)
Denoising Autoencoders learn to reconstruct clean data from corrupted inputs, forcing the model to capture meaningful structure in the data distribution rather than simply memorizing inputs @bengio_generalized_2013. By training on progressively noisier samples, DAEs learn a mapping that moves samples toward higher-density regions of the data manifold. This denoising objective forms the conceptual foundation for diffusion-based generative models, where generation is performed through iterative noise removal.

=== Diffusion Models
Diffusion models define a generative process as the reversal of a gradual noising procedure applied to training data @ho_denoising_2020. During training, noise is incrementally added to data over a sequence of timesteps, while the model learns to predict and remove this noise. The general architecture used for diffusion models are Unets @salimans_pixelcnn_2017 which are a natural choice to map corrupted data to reverse process parameters. At inference time, the model generates samples by starting from pure noise and iteratively denoising, producing high-quality and diverse images with stable training dynamics.

=== ResNet
Residual Networks introduce skip connections that allow information to bypass intermediate layers, mitigating issues such as vanishing gradients in deep neural networks. By learning residual functions rather than direct mappings, ResNets enable more stable optimization and improved feature propagation @thakur_efficient_2023. These properties make them well-suited for generative models, where preserving low-level spatial information across layers is critical.

== Upsampler Models
=== PixelShuffle
PixelShuffle is an efficient upsampling operation that rearranges channel information into spatial resolution, increasing image size without introducing checkerboard artifacts common in transposed convolutions @zamzam_pixelshuffler_2025. This approach enables computationally efficient super-resolution while maintaining spatial consistency in generated images.

== Refiner Models

=== Vision Transformers
Vision Transformers (ViTs) model images as sequences of patches and use self-attention to capture long-range dependencies across the entire image @dosovitskiy_image_2021. This global receptive field allows refiners to reason about spatial relationships and semantic consistency, making ViTs particularly effective for correcting structural inconsistencies and improving perceptual coherence.

== Cascaded diffusion models 
Cascaded generative models approach image synthesis by breaking the generation process into multiple sequential stages, each operating at an increasing level of resolution and detail @ho_cascaded_2021. Prior work has shown that generating images in this progressive manner simplifies the learning problem, as early stages focus on global structure while later stages specialize in adding fine-grained details. In the context of diffusion models, cascades typically consist of an initial low-resolution generator followed by learned super-resolution models that condition on the outputs of previous stages. More broadly, cascaded approaches highlight how complex image distributions can be approximated through a composition of simpler models rather than a single, highly complex network, providing both practical performance benefits and increased interpretability.

This is an approach with merit but is prone to issues such as error propagation throughout the pipeline. There are many ways in which this has been approached, teacher forcing, conditional augmentation @ho_cascaded_2021 to name a few are approaches used to manage the loss carrying from one stage to the next during training. While results have been shown for cascaded models there is much room for growth and exploration such as alternative models that only diffusion which would allow more diverse methods of training.
#pagebreak()

= Method

== Dataset and task
#v(1em)
Our project uses the CIFAR-10 dataset, which consists of 60,000 training images of size 32x32 across 10 object categories. In this project, the labels are ignored, and the task is image generation. Images are loaded and converted to tensors in (0,1). No significant augmentation is applied, as the quality of generative images is the primary concern rather than classification performance. 

CIFAR-10 was the chosen dataset as it allows quick generation due to the 32x32 size, it is also a heavily tested framework which allows for easier comparisons with the current state of the art.
#v(1em)
== Cascaded Model Training Schedule
#v(1em)
The generator is implemented as a three stage pipeline, which base input is a 16x16 noisy image:

- The generation stage: 16x16 coarse image
- The upsampler stage: 32x32 upsampled image
- The refiner stage: 32x32 refined image

All stages inherit from Baseclass, which enforces they implement preprocess() and forward(). These functions are essential to be able to properly handle the different models as each model handles inputs and training differently.This abstraction makes it possible to easily switch the different underlying models (like changing from a simple convolutional refiner to ViT refiner).

#v(1em)
== Training
#v(1em)
The training has three stages:

=== Stage 1: Coarse Downsampled Image Generation
The first stage of training is the "warm up" stage for the coarse model. This is required as the following refiner and upsampler layers are based on the output and the early learning outputs would propagate errors throughout the pipeline. The loss function for the generation stage is MSE.

#v(1em)

=== Stage 2: Upsampler and Refiner

In stage 2, coarse generation stage is frozen and the upsampler and refiner models are trained. Due to the unstable nature of the generation stage a teacher forcing schedule is used.
This decides if the refiner and upsampler get a ground truth downsampled image or the coarse stage output. 
The scheduler for this lowers the possibility of giving the ground truth to the upsampler linearly over the number of stage 2 epochs. The start and end points are configurable.
The losses in this phase include: an MSE loss for the upsampler output compared to a resized ground truth image, a refiner loss that is comprised of MSE and a perception loss.
#v(1em)

=== Stage 3: Fine Tuning
During the last stage, all three stages get unfrozen and are trained together with the same loss combinations.
In this stage the gradients are not detached from the coarse stage allowing the gradients to flow from the upsampler and refiner to the coarse model to allow for more fine tuned training of the weights to work in tandem with the cascading models.

=== Stage Information

Each phase is assigned its optimizer and learning rate scheduler, the settings are made through a configuration. The Config class is responsible for defining the core hyperparameters, the number of layers in the coarse block, the dimensions and heights of the upsampler and refiner, and the teacher forcing start and end probabilities. 
This configuration was designed with ablations studies in mind to allow for comprehensive testing and comparison.
#v(1em)

= Experimental analysis and evaluation

== Baseline configuration
The baseline configuration for all empirical tests is: 
#table(
  columns: (auto, auto),
  inset: 6pt,
  align: (left, left),
  stroke: 0.6pt,

  [*Category*], [*Configuration*],

  [Dataset],
  [
    *Batch size:* 128
  ],

  [Diffusion Parameters],
  [
    *Timesteps (T):* 25 \\
    *Beta start:* 1e-4 \\
    *Beta end:* 0.02 \\
    *Timestep embedding dim:* 32
    *Time embedding:* 128
  ],

  [Coarse Generator],
  [
    *Resolution:* 16 x 16 \\
    *Model type:* Diffusion \\
    *Input channels:* 3 \\
    *Output channels:* 128 \\
    *Latent dimension:* 256 \\
    *Layers:* 6 \\
    *Skip connection every:* 2 layers \\
    *DAE sigma:* 0.5 \\
  ],

  [Upsampler],
  [
    *Model type:* PixelShuffle \\
    *Input channels:* 3 \\
    *Hidden channels:* 256 \\
    *Upscale factor:* 2 \\
    *Residual blocks:* 2
  ],

  [Refiner],
  [
    *Model type:* Vision Transformer (ViT) \\
    *Input channels:* 3 \\
    *Feature channels:* 256 \\
    *Residual blocks:* 2
  ],

  [Training Schedule],
  [
    *Coarse epochs:* 35 \\
    *Upsampler + Refiner epochs:* 35 \\
    *Joint training epochs:* 30
  ],

  [Optimization],
  [
    *Learning rate (all stages):* 2e-4 \\
    *LR step size:* 15 \\
    *LR decay (gamma):* 0.5
  ],

  [Teacher Forcing],
  [
    *Phase 2:* 0.9 → 0.1 \\
    *Phase 3:* 0.1 → 0.0
  ],
)

These were selected based on earlier tests and served as a baseline for all future tests

== Ablation studies
The goal of the ablation study was to identify if there was a correlation between parameter count and performance and what combination of cascading models produce the best results.

To conduct fair tests the only features that were altered between tests were, the models used in the cascading Ex from diffusion -> resnet ->  vit to dae -> pixel shuffle -> vit.
These were selected to allow a diverse range of models to be tested and to explore the correlation between parameter count and models between different cascades.

== Evaluation metrics

=== Kernel Inception Distance (KID)
#v(1em)
KID quantifies the correspondence of the generated images to the real CIFAR-10 images in the Inception network's feature space by means of their distributions. A lower value of KID indicates that the generated samples are, on average, even closer to the real image, as far as the visual features learnt are concerned. In our case, KID is calculated regularly throughout the training period by using a constant number of real and generated samples, and we present both the final and mean KID over the recorded assessments.

=== LPIPS (perceptual distance)
#v(1em)
LPIPS uses deep features (VGG) instead of raw pixel differences to evaluate the similarity between the two images. It uses a pretrained model to calculate the perception loss. This is useful because MSE favours average values but that leads to heavy blur and homogeneous colours over sharp edged. In our processing, LPIPS is incorporated into the refiner loss alongside MSE, encouraging outputs that are not only numerically close to the target but also visually coherent.

=== Time per epoch
#v(1em)
The duration of the epoch is recorded to capture computational cost and training efficiency. This helps put improvements in KID and visual quality into perspective against training time, which is important when model size or diffusion schedule is varied across configurations.

== Results

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 6pt,
  align: (left, left, left, left, left),
  stroke: 0.6pt,

  [*Configuration name*], [*Total params*], [*Number layers*], [*Final KID*], [*Total mean loss*],

  [Diffusion + PixelShuffle + ViT], [8466441], [55], [0.166], [0.1649],
  [Diffusion + ResNet + ViT],       [9058569], [63], [0.171], [0.1644],
  [Diffusion + Linear + ViT],       [6104073], [47], [0.151], [0.1612],

  [ResNet + ResNet + ViT],          [8731785], [61], [0.153], [0.09175],
  [ResNet + PixelShuffle + ViT],    [8139657], [53], [0.159], [0.09177],
  [ResNet + Linear + ViT],          [5777289], [45], [0.150], [0.0891],

  [DAE + ResNet + ViT],             [8731785], [61], [0.165], [0.09202],
  [DAE + PixelShuffle + ViT],       [8139657], [53], [0.146], [0.09204],
  [DAE + Linear + ViT],             [5777289], [45], [0.172], [0.0886],
)

=== KID across configurations
#image("kid_across_configs.png", width: 80%)
=== Layer count across configurations
#image("LayerCountvsKID.png", width: 80%)
=== Parameter count vs KID
#image("ParametercountvsKID.png", width: 80%)

*Labels:*

- *Config 0* - Diffusion + PixelShuffle + ViT  
- *Config 1* - Diffusion + ResNet + ViT  
- *Config 2* - Diffusion + Simple Upsampler + ViT  

- *Config 3* - ResNet + ResNet + ViT  
- *Config 4* - ResNet + PixelShuffle + ViT  
- *Config 5* - ResNet + Simple Upsampler + ViT  

- *Config 6* - Denoising Autoencoder + ResNet + ViT  
- *Config 7* - Denoising Autoencoder + PixelShuffle + ViT  
- *Config 8* - Denoising Autoencoder + Simple Upsampler + ViT  


#pagebreak()

= Conclusion
In light of the results it is clear that there is no strong bearing on the types of models used in sequence to generate images. This suggests that although there have been methods that effectively use cascading models to great effect @ho_cascaded_2021, there is a need for specific integration methods between models rather than an abstract stage based approach.
When each stage is defined abstractly, there can only be a weak abstract connection between each model reducing the effectiveness of the approach. This paper highlights shows that while there is merit in cascading models, there is a need for care when selecting each models to ensure they connect and flow well from one to another sharing gradients to allow for more accurate training.

Overall, this work shows that cascaded generative models can serve as a viable and flexible framework for image synthesis, but their effectiveness depends on thoughtful architectural and training choices. Future improvements are likely to come from tighter stage integration and more principled design rather than simply increasing model complexity.




#pagebreak()

= References
#bibliography("references.bib")

