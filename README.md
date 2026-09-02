# EquiFusion - Kinematics-Agnostic Human Motion Prediction via Equivariant Latent Diffusion (ECCV 2026)
**[Website](https://ceveloper.github.io/publications/equifusion/)** |
**[Paper](https://arxiv.org/abs/2607.10984)**

This is the official repository for the ECCV 2026 paper
"_EquiFusion: Kinematics-Agnostic Human Motion Prediction via Equivariant Latent Diffusion_".

Authors:
[Cecilia Curreli](https://ceveloper.github.io/),
[Florian Hofherr](https://florianhofherr.github.io/),
[Dominik Muhle](https://dominikmuhle.github.io/),
[Abhishek Saroha](https://cvg.cit.tum.de/members/saroha),
[Riccardo Marin](https://ricma.netlify.app/),
[Daniel Cremers](https://cvg.cit.tum.de/members/cremers)

<p align="center">
  <img src="./figures/teaser_animated_social.gif" alt="EquiFusion Teaser" width="600"/>
</p>

> ### 🚧 Code coming soon
> The model and training code for **EquiFusion** will be released here. **Star** or **watch** this repository to be notified when it lands. In the meantime, see the [paper](https://arxiv.org/abs/2607.10984) and the [project page](https://ceveloper.github.io/publications/equifusion/).

## Intro

We address **kinematics-agnostic** (skeleton-agnostic) **stochastic 3D human motion prediction**: given a past motion, predict plausible futures for *any* skeleton, without hard-coding its kinematics. EquiFusion is a **permutation-equivariant latent diffusion** model that treats the skeleton's kinematic connectivity as an explicit input, so its internal computations are agnostic to joint ordering and graph structure.

This design enables **cross-dataset / cross-skeleton generalization to unseen kinematics**, and unlocks **zero-shot** directions such as **motion prediction from partial or occluded observations** and **targeted limb generation** — while remaining up to **75% more compact** than kinematics-specific baselines, with faster training and inference.

<!-- Optional: paste the official abstract here (your own CC BY 4.0 text). The paragraphs above
     are written to front-load search keywords for cold discovery by humans and agents. -->

<!-- ### Index -->

<!-- Uncomment / complete as sections are added when the code is released.
- [Intro](#intro)
- [Installation](#installation)
- [Datasets](#datasets)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Training](#training)
- [Visualizations](#visualizations)
- [License](#license)
- [Citation](#citation)
-->

<!-- ## Results

State-of-the-art on standard stochastic 3D human motion prediction benchmarks. -->

<!-- TODO: name the exact benchmarks/datasets you report on (e.g. Human3.6M, AMASS, ...) and add
     the results table. Naming them explicitly helps human and agent search find this repo by
     problem rather than by method name. -->

<!-- | Benchmark | Metric | EquiFusion |
| --------- | ------ | ---------- |
| _TBD_     | _TBD_  | _TBD_      | -->

## License

This code is released under the **BSD 2-Clause License**. See [LICENSE](LICENSE) for details.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{curreli2026equifusion,
  title     = {EquiFusion: Kinematics-Agnostic Human Motion Prediction via Equivariant Latent Diffusion},
  author    = {Curreli, Cecilia and Hofherr, Florian and Muhle, Dominik and Saroha, Abhishek and Marin, Riccardo and Cremers, Daniel},
  booktitle = {European Conference on Computer Vision},
  organization = {Springer},
  year      = {2026}
}
```