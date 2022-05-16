# Abstract及其翻译

[toc]

## 1.原文:

<font size = 5>**Differentiable Augmentation for Data-Efficient GAN Training**</font>
<p align = right> From NeurIPS </p>

<font size = 4> **[1]** The performance of generative adversarial networks (GANs) heavily deteriorates given a limited amount of training data. **[2]** This is mainly because the discriminator is memorizing the exact training set. **[3]** To combat it, we propose Differentiable Augmentation (DiffAugment), a simple method that improves the data efficiency of GANs by imposing various types of differentiable augmentations on both real and fake samples. **[4]** Previous attempts to directly augment the training data manipulate the distribution of real images, yielding little benefit; DiffAugment enables us to adopt the differentiable augmentation for the generated samples, effectively stabilizes training, and leads to better convergence. **[5]** Experiments demonstrate consistent gains of our method over a variety of GAN architectures and loss functions for both unconditional and class-conditional generation. **[6]** With DiffAugment, we achieve a state-of-the-art FID of 6.80 with an IS of 100.8 on ImageNet 128x128 and 2-4x reductions of FID given 1,000 images on FFHQ and LSUN. **[7]** Furthermore, with only 20% training data, we can match the top performance on CIFAR-10 and CIFAR-100. **[8]** Finally, our method can generate high-fidelity images using only 100 images without pre-training, while being on par with existing transfer learning algorithms. **[9]** Code is available at https://github.com/mit-han-lab/data-efficient-gans.</font>

## 2.翻译:

### 标题:

原文：**Differentiable Augmentation for Data-Efficient GAN Training**
意思为**数据高效型GAN训练的可微增广方法**
其中，GAN意思为**Generative Adversarial Networks**,生成式对抗网路，一种深度学习模型

### 具体内容
* **[1]** 在训练数据有限的情况下，生成式对抗网络(GANs)的性能严重恶化。

* **[2]** 这主要是因为鉴别器记忆的是准确的训练集。

* **[3]** 为了解决这个问题，我们提出了**可微增宽(Differentiable Augmentation)** 方法，这是一种简单的方法，通过在真实和虚假样本上施加各种类型的可微增宽来提高GANs的数据效率。

* **[4]** 以前直接增加训练数据的尝试是控制真实图像的分布，**收效甚微**;
DiffAugment使我们可以对生成的样本采用可微的增强，**有效地稳定训练，使收敛性更好**。

* **[5]** 实验表明，我们的方法在各种GAN架构和无条件和类条件生成的损失函数上都取得了**一致的收益**。
* **[6]** 使用DiffAugment技术，我们在ImageNet 128x128上实现了最先进的FID为6.80, IS为100.8，在FFHQ和LSUN上给1000张图像时实现了2-4倍的FID缩减。
* **[7]** 此外，只需20%的训练数据，我们就可以媲美CIFAR-10和CIFAR-100的顶级性能。
* **[8]** 最后，我们的方法可以在不进行预处理的情况下，只使用**100张图像便可以生成高保真图像**，与现有的迁移学习算法相当。
* **[9]** 代码可以在https://github.com/mit-han-lab/data-efficient-gans上找到。