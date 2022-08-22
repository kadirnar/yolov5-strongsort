<div align="center">
<h1>
  Yolov5-StrongSort: YOLOv5-Pip + StrongSort
</h1>

<h4>
    <img width="700" alt="teaser" src="doc/uav.gif">
</h4>
</div>

## <div align="center">Overview</div>

This repo is a shortened version of yolov5 codes and added StrongSort algorithm.

### Installation

```
git clone https://github.com/kadirnar/yolov5-strongsort
cd yolov5-strongsort
pip install -r requirements.txt
```

### Yolov5 Model + StrongSort Prediction

```
python yolo_tracker.py --video_path test.mp4 --model_path yolov5m.pt --device cuda:0 --confidence_threshold 0.5 --image_size 640 --config_path osnet_x0_25_market1501.pt
```

## Citations
```bibtex
@article{du2022strongsort,
  title={Strongsort: Make deepsort great again},
  author={Du, Yunhao and Song, Yang and Yang, Bo and Zhao, Yanyun},
  journal={arXiv preprint arXiv:2202.13514},
  year={2022}
}
```

### Reference:
 - [Yolov5-Pip](https://github.com/fcakyon/yolov5-pip)
 - [Strongsort](https://github.com/dyhBUPT/StrongSORT)
