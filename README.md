<h2 align="center">Class Prior-Free Positive-Unlabeled Learning with Taylor Variational Loss for Hyperspectral Remote Sensing Imagery</h2>


<h5 align="right">
by <a href="https://hengwei-zhao96.github.io">Hengwei Zhao</a>,
Xinyu Wang,
and <a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>
</h5>

[[`arXiv`](https://arxiv.org/abs/2308.15081)]
[[`Paper(ICCV 2023)`](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhao_Class_Prior-Free_Positive-Unlabeled_Learning_with_Taylor_Variational_Loss_for_Hyperspectral_ICCV_2023_paper.pdf)]

---------------------

This is an official implementation of _T-HOneCls_ in our ICCV 2023 paper.

## Highlights:
1. Class prior-free PU learning for limited labeled hyperspectral imagery
2. _T-HOneCls_ achieves state-of-the-art results on 7 datasets (21 tasks in total)

## Requirements:
- pytorch >= 1.13.1
- GDAL ==3.4.1

## Running:
1.Modify the data path in the configuration file (./configs/X/XX/XXX.py).
The hyperspectral data can be obtained from the [`Link`](https://pan.baidu.com/s/1Ac3ko3BcZ4sS_cmzZhA7ow?pwd=sqyy )(password:sqyy)

2.Training and testing
```bash
sh scripts/HongHu.sh
sh scripts/LongKou.sh
sh scripts/HanChuan.sh
```

## Citation:
If you use _T-HOneCls_ in your research, please cite the following paper:
```text
@InProceedings{Zhao_2023_ICCV,
    author    = {Zhao, Hengwei and Wang, Xinyu and Li, Jingtao and Zhong, Yanfei},
    title     = {Class Prior-Free Positive-Unlabeled Learning with Taylor Variational Loss for Hyperspectral Remote Sensing Imagery},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {16827-16836}}

@article{ZHAO2022328,
    title = {Mapping the distribution of invasive tree species using deep one-class classification in the tropical montane landscape of Kenya},
    journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
    volume = {187},
    pages = {328-344},
    year = {2022},
    issn = {0924-2716},
    doi = {https://doi.org/10.1016/j.isprsjprs.2022.03.005},
    url = {https://www.sciencedirect.com/science/article/pii/S0924271622000715},
    author = {Hengwei Zhao and Yanfei Zhong and Xinyu Wang and Xin Hu and Chang Luo and Mark Boitt and Rami Piiroinen and Liangpei Zhang and Janne Heiskanen and Petri Pellikka}}

@ARTICLE{10174705,
    author={Zhao, Hengwei and Zhong, Yanfei and Wang, Xinyu and Shu, Hong},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={One-Class Risk Estimation for One-Class Hyperspectral Image Classification}, 
    year={2023},
    volume={},
    number={},
    pages={1-1},
    doi={10.1109/TGRS.2023.3292929}}
```
_T-HOneCls_ can be used for academic purposes only, and any commercial use is prohibited.
<a rel="license" href="https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en">

<img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a>