<div align="center">
<h1>
  Yolov5-Lite: Minimal YOLOv5 + Deep Sort
</h1>

<h4>
    <img width="700" alt="teaser" src="resources/uav.gif">
</h4>
</div>

## <div align="center">Overview</div>

This repo is a shortened version of yolov5 codes and added deep sort algorithm.

### Installation

```
git clone https://github.com/kadirnar/yolov5-lite
cd yolov5-lite
pip install -r requirements.txt
```

### Yolov5 Model + Deep Sort Prediction

```
python detect.py --source ../uav_videos/3.mp4 --show-vid --classes  4 --device 0
```

### Reference:

 - [YOLOv5](https://github.com/ultralytics/yolov5)
 - [Deep Sort](https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch)

## Citations
```bibtex
@misc{yolov5deepsort2020,
    title={Real-time multi-object tracker using YOLOv5 and deep sort},
    author={Mikel Broström},
    howpublished = {\url{https://github.com/mikel-brostrom/Yolov5_DeepSort_Pytorch}},
    year={2020}
}
