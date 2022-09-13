from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import LoadImages
from pathlib import Path
import argparse
import torch 
import cv2
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from strongsort.strong_sort import StrongSORT
class Yolov5Sort:
    def __init__(
        self,
        model_path: str = "yolov5m.pt",
        config_path: str = 'osnet_x0_25_market1501.pt',
        device: str = "cpu",
        confidence_threshold: float = 0.5,
        image_size: int = 640,
        view_img: bool = True,
        augment: bool = False,
        save_video: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.load_model()
        self.prediction_list = None
        self.image_size = image_size
        self.config_path = config_path
        self.view_img = view_img
        self.augment = augment
        self.save_video = save_video
        
    def load_model(self):
        import yolov5

        model = yolov5.load(self.model_path, device=self.device)
        model.conf = self.confidence_threshold
        self.model = model
    
    def yolo_tracker(self, video_path):
        dataset = LoadImages(video_path, self.image_size)  
        strongsort_list = []        
        strongsort_list.append(
            StrongSORT(
                model_weights= self.config_path,
                device=select_device(self.device),
            )
        )
        
        outputs = [None] 
          
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred = self.model(im, size=self.image_size, augment=self.augment) 
            pred = non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=self.model.iou, classes=self.model.classes, agnostic=self.model.agnostic)

            for i, det in enumerate(pred):
                annotator = Annotator(im0s, line_width=2, example=str(self.model.names))

                if len(det):
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
                    xywhs = xyxy2xywh(det[:, 0:4]).cpu().detach().numpy()
                    confs = det[:, 4].cpu().detach().numpy()
                    clss = det[:, 5].cpu().detach().numpy()
                    outputs[i] = strongsort_list[i].update(xywhs, confs, clss, im0s)
                    if len(outputs[i]) > 0:
                        for j, (output, conf) in enumerate(zip(outputs[i], confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            
                            if self.view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                id = int(id)  # integer id
                                label = label = "%s %.2f" % (self.model.names[int(cls)], conf)
                                annotator.box_label(bboxes, label, color=colors(c, True))
            # Stream results
            im0 = annotator.result()
            if self.view_img:
                cv2.imshow(str(path), im0)
                cv2.waitKey(1)  # 1 millisecond


def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO v5 video stream detector')
    parser.add_argument('--model_path', type=str, default='yolov5m.pt', help='path to weights file')
    parser.add_argument('--config_path', type=str, default='osnet_x0_25_market1501.pt', help='path to configuration file')
    parser.add_argument('--image_size', type=int, default=640, help='size of each image dimension')
    parser.add_argument('--video_path', type=str, default='test.mp4', help='path to input video file')
    parser.add_argument('--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--augment', action='store_true', help='augmented video')

    return parser.parse_args()

def run(args):
    Yolov5Sort(args.model_path, args.config_path, args.device, args.confidence, args.image_size, args.view_img, args.augment).yolo_tracker(args.video_path)

if __name__ == '__main__':
    args = parse_arguments()
    run(args)
    