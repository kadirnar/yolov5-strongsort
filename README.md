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
Yolov5Sort(
  model_path='yolov5x.pt', 
  device='cuda:0', 
  confidence_threshold=0.5, 
  image_size=640, 
  config_path='osnet_x0_25_market1501.pt').yolo_tracker('test.mp4')
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
```bibtex
@misc{yolov5-strongsort-osnet-2022,
    title={Real-time multi-camera multi-object tracker using YOLOv5 and StrongSORT with OSNet},
    author={Mikel Broström},
    howpublished = {\url{https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet}},
    year={2022}
}
```

### Reference:
 - [YOLOv5](https://github.com/ultralytics/yolov5)
 - [Strongsort](https://github.com/dyhBUPT/StrongSORT)
