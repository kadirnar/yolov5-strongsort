from ast import arg
from yolov5.utils.general import non_max_suppression, scale_coords, xyxy2xywh, increment_path
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


from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT


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
        project=ROOT / 'runs/track'
        save_dir = increment_path(Path(project) /  "exp", exist_ok=False)  # increment run
        (save_dir / 'tracks' if False else save_dir).mkdir(parents=True, exist_ok=True)  # make dir 
        cfg = get_config()
        cfg.merge_from_file('strong_sort/configs/strong_sort.yaml')

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
        vid_path, vid_writer = [None] * nr_sources, [None] * nr_sources
        curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources   
              
        for path, im, im0s, vid_cap, s in dataset:
            im = torch.from_numpy(im).to(self.device)
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

            pred = self.model(im, size=self.image_size, augment=self.augment) 
            pred = non_max_suppression(pred, conf_thres=self.model.conf, iou_thres=self.model.iou, classes=self.model.classes, agnostic=self.model.agnostic)
            save_path = str(save_dir / Path(path).name)

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

            # Save results (image with detections)
            if self.save_video:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]


def parse_arguments():
    parser = argparse.ArgumentParser(description='YOLO v5 video stream detector')
    parser.add_argument('--model_path', type=str, default='yolov5m.pt', help='path to weights file')
    parser.add_argument('--config_path', type=str, default='osnet_x0_25_market1501.pt', help='path to configuration file')
    parser.add_argument('--image_size', type=int, default=640, help='size of each image dimension')
    parser.add_argument('--video_path', type=str, default='test.mp4', help='path to input video file')
    parser.add_argument('--confidence', type=float, default=0.5, help='minimum probability to filter weak detections')
    parser.add_argument('--device', default='cpu', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--save_video', action='store_true', help='save results video')    
    parser.add_argument('--view_img', action='store_true', help='display results')
    parser.add_argument('--augment', action='store_true', help='augmented video')

    return parser.parse_args()

def run(args):
    Yolov5Sort(args.model_path, args.config_path, args.device, args.confidence, args.image_size, args.view_img, args.augment, args.save_video).yolo_tracker(args.video_path)

if __name__ == '__main__':
    args = parse_arguments()
    run(args)
