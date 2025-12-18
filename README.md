# StegaVAR

Official implementation of **StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis**. This paper has been accepted by **AAAI 2026**.

![intro](assets/fig1_cr.pdf)

## Data Preprocessing

Before training, you need to extract frames from the video datasets (e.g., HMDB51, UCF101). We provide a utility script `src/utils/video.py` for this purpose.

1. Organize your dataset as follows:
   ```
   dataset/VAR/hmdb51/hmdb51/
       brush_hair/
           video1.avi
           video2.avi
       ...
   ```
2. Run the preprocessing script:
   ```bash
   python src/utils/video.py
   ```
   This script will extract frames from each `.avi` file and save them into a corresponding directory.

## Training Examples

### Training without Steganography

To train the model for VAR without any steganography:

```bash
python main.py --run_id ucf101_nohide --task har --model r3dpro_time_ta
```

### Training with Steganography

To train with steganography enabled, use the `--hide` flag. 

Example command for training with steganography:

```bash
python main.py --hide --run_id ucf101_hide --task har --model r3dpro_time_ta --hide_model lfvsn
```

You can also specify different steganography models using the `--hide_model` argument (e.g., `hinet`, `lfvsn`, `wengnet`, `hidden`).

## Acknowledgements

This project is compatible with the steganography implementations from the following repositories:
- [Hiding-images-within-images](https://github.com/albblgb/Hiding-images-within-images)
- [LF-VSN](https://github.com/MC-E/LF-VSN)

## BibTeX

```bibtex
@misc{chen2025stegavarprivacypreservingvideoaction,
      title={StegaVAR: Privacy-Preserving Video Action Recognition via Steganographic Domain Analysis}, 
      author={Lixin Chen and Chaomeng Chen and Jiale Zhou and Zhijian Wu and Xun Lin},
      year={2025},
      eprint={2512.12586},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.12586}, 
}
```
