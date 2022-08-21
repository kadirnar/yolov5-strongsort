
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from yolov5.utils.plots import Annotator, colors
from yolov5.utils.torch_utils import select_device
from yolov5.utils.dataloaders import LoadImages
import torch 
import cv2
import os

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


class Yolov5Sort:
    def __init__(
        self,
        model_path: str,
        device: str,
        confidence_threshold: float = 0.5,
        image_size: int = 640,
        config_path: str = 'osnet_x0_25_market1501.pt',
        view_img: bool = True,
        augment: bool = False,
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
        
    def load_model(self):
        import yolov5

        model = yolov5.load(self.model_path, device=self.device)
        model.conf = self.confidence_threshold
        self.model = model
    
    def yolo_tracker(self, video_path):
        dataset = LoadImages(video_path, self.image_size)   
        # initialize StrongSORT
        cfg = get_config()
        cfg.merge_from_file('strong_sort/configs/strong_sort.yaml')

        # Create as many strong sort instances as there are video sources
        strongsort_list = []
        nr_sources=1
        for i in range(nr_sources):
            strongsort_list.append(
                StrongSORT(
                    model_weights= self.config_path,
                    device=select_device(self.device),
                    fp16 = False,
                    max_dist=cfg.STRONGSORT.MAX_DIST,
                    max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.STRONGSORT.MAX_AGE,
                    n_init=cfg.STRONGSORT.N_INIT,
                    nn_budget=cfg.STRONGSORT.NN_BUDGET,
                    mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                    ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

                )
            )
            strongsort_list[i].model.warmup()
        outputs = [None] * nr_sources
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources         
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred = self.model(im, size=self.image_size, augment=self.augment) 
            pred = non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=self.model.iou, classes=self.model.classes, agnostic=self.model.agnostic)

            for i, det in enumerate(pred):
                curr_frames[i] = im0s
                annotator = Annotator(im0s, line_width=2, example=str(self.model.names))
                if cfg.STRONGSORT.ECC:  # camera motion compensation
                    strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

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
                
                prev_frames[i] = curr_frames[i]
