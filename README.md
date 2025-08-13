
<div align="center">

# Markov Chain Estimation with In-Context Learning

<a href="https://simon-lepage.github.io"><strong>Simon Lepage</strong></a>
—
<strong>Jeremie Mary</strong>
—
<a href=https://davidpicard.github.io><strong>David Picard</strong></a>

<a href=https://ailab.criteo.com>CRITEO AI Lab</a>
&
<a href=https://imagine-lab.enpc.fr>ENPC</a>
</div>

<p align="center">
    <a href="https://arxiv.org/abs/2508.03934">
        <img alt="ArXiV Badge" src="https://img.shields.io/badge/arXiv-2508.03934-b31b1b.svg">
    </a>
</p>

## Installation 

Create a new python environments (code written with 3.9.19) and install the requirements:
```shell
pip install -r requirements.txt
```

## Usage

We use [`hydra`](https://hydra.cc/docs/intro/) to configure our experiments. The configuration files are in the `config/` folder. You can easily override parameters through the CLI.

```shell
python ./main.py experiment=default # ~10GB VRAM
python ./main.py experiment=permutation # ~10GB VRAM
python ./main.py experiment=ortho # ~36GB VRAM
```

**About orthogonal encoding:** Convergence is slower for the orthogonal encoding scheme, and the loss tends to show multiple long plateaus with small drops. Do not hesitate to train longer, even if you don't see immediate improvements. I typically see drops around 8k and 12k steps.

## Citation

To cite our work, please use the following BibTeX entry : 
```bibtex
@article{lepage2025markov,
  title={Markov Chain Estimation with In-Context Learning},
  author={Lepage, Simon and Mary, Jeremie and Picard, David},
  journal={GRETSI},
  year={2025}
}
```
