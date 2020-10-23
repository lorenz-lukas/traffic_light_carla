# python3 main_predict.py --source input/videos/t5_5.mp4 --save-txt --view-img --weights weights/best_model_3.pt --cfg cfg/labels/yolov3-spp-2cls.cfg --names data/traffic_light_2.names


import argparse
import sys
sys.path.append('../')

from libs.models import *  # set ONNX_EXPORT in models.py
from libs.utils.datasets import *
from libs.utils.utils import *
import cv2
from json import dumps

class Yolov3():
    model_classifier = None
    model = None
    t0 = 0
    video_cap = None
    video_writer = None
    dataset = None
    img = None
    video_path = None
    save_flag = None
    nframes = 1 # At least one frame

    classify = False
    view_image = True
    exit_key = False

    data_result = {"Img": None, "Label": None}
    list_result = []
    outpath = None
    FR = 33 #Frame rate
    
    def __init__(self, source = None):
        self.cfg = '../model/cfg/yolov3-spp-2cls.cfg'
        self.weights = '../model/weights/best_model_3.pt'
        self.classes = None
        self.classes_name = load_classes('../model/data/traffic_light_2.names')
        self.colors = [(0, 255, 0), (0, 0, 255), (0, 0, 155), (0, 200, 200), (29, 118, 255), (0 , 118, 255)]
        
        if(source):
            self.source = source ## CAMERA ID OR VIDEO/IMAGE
        else: 
            self.source = 'test_video_data/carla.mp4'
        
        self.img_size = 608
        self.iou_threshold = 0.6
        self.conf_threshold = 0.3
        self.outputDIR = 'outputs'
        self.device = ''
        self.video_format = 'mp4v'
        self.agnostic = False
        self.augmented = False
        self.half = False
        self.show_img = True
        self.save_results = True

    def __del__(self):
        if(self.video_cap):
            self.video_cap.release()
        if(self.video_writer):
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("EXITING!")
    
    def setConfigs(self):
        self.cfg = opt.cfg
        self.outputs = opt.output
        self.img_size = opt.img_size
        self.source = opt.source ## CAMERA ID OR VIDEO/IMAGE
        self.weights = opt.weights
        self.half = opt.half
        self.show_img = opt.view_img
        self.save = opt.save_txt

        self.device = opt.device
        self.classes_name = load_classes(opt.names)
        
        self.augmented = opt.augment
        self.video_format = opt.fourcc
        self.iou_threshold = opt.iou_thres
        self.conf_threshold = opt.conf_thres
        self.agnostic = opt.agnostic_nms

    def loadWeights(self):
        attempt_download(self.weights)
        if self.weights.endswith('.pt'):  # pytorch format
            self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
        else:  # darknet format
            load_darknet_weights(self.model, self.weights)
    
    def runInference(self):
        self.t0 = time.time()
        self.img = torch.zeros((1, 3, self.img_size, self.img_size), device=self.device)  # init img
        _ = self.model(self.img.half() if self.half else self.img.float()) if self.device.type != 'cpu' else None  # run once

    def cassify(self):
        self.model_classifier = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        self.model_classifier.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        self.model_classifier.to(self.device).eval()

    def loadDataset(self):
        if(self.source.isnumeric()):
            self.show_img = True
            torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
            self.dataset = LoadStreams(self.source, img_size=self.img_size)
            print("\n[WANING] Using camera stream!\n\n")
        else:
            self.save = True
            self.dataset = LoadImages(self.source, img_size=self.img_size)
            print("\n[WARNING] Using preload dataset!\n\n")
    
    def setNetwork(self):
        self.img_size = (320, 192) if ONNX_EXPORT else opt.img_size
        self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else self.device)
        
        self.model = Darknet(self.cfg, self.img_size)
        # load weights 
        self.loadWeights()
        self.model.to(self.device).eval()
        # Export mode
        if(ONNX_EXPORT):
            self.model.fuse()
            img = torch.zeros((1, 3) + self.img_size)  # (1, 3, 320, 192)
            f = self.weights.replace(self.weights.split('.')[-1], 'onnx')  # *.onnx filename
            torch.onnx.export(self.model, img, f, verbose=False, opset_version=11,
                            input_names=['images'], output_names=['classes', 'boxes'])

            # Validate exported model
            import onnx
            self.model = onnx.load(f)  # Load the ONNX model
            onnx.checker.check_model(self.model)  # Check that the IR is well formed
            print(onnx.helper.printable_graph(self.model.graph))  # Print a human readable representation of the graph
            return
        # Half precision
        self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
        if self.half:
            self.model.half()
        # Set Dataloader
        self.loadDataset()
        # Run inference   
        self.runInference()

    def saveResults(self, p):
        if(self.dataset.mode == 'images'):
            cv2.imwrite(self.video_path, self.img)
        else:
            if(self.video_path != self.save_flag):  # new video
                self.save_flag = self.video_path            
                if isinstance(self.video_writer, cv2.VideoWriter):
                    self.video_writer.release()  # release previous video writer

                fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                w = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_writer = cv2.VideoWriter(self.video_path, cv2.VideoWriter_fourcc(*self.video_format), fps, (w, h))
                
            self.video_writer.write(self.img)
            cv2.imwrite(self.outpath, self.img)

    def showResults(self, p = "Frame", j = 0, label = ""):
        # Stream results
        cv2.imshow(p, self.img)
        if cv2.waitKey(self.FR) == ord('q'):
            #print(f"Average FPS: {i/(time.time() - self.t0)}")
            self.exit_key = True

        if(self.save_results):
            self.list_result.append({"Img": self.outpath, "Label": label})
            self.saveResults(j)        

        # if cv2.waitKey(1) & 0xFF == ord('q'):  # q to quit
        #     print(f"Average FPS: {i/(time.time() - self.t0)}")
        #     self.exit_key = True
        #     #raise Exception("Exiting!")


    def evaluate_frame(self, im0s = None, img = None, j = 0, path = "frame.png"):
        # Inference
        t1 = torch_utils.time_synchronized()
        pred = self.model(img, augment=self.augmented)[0]
        t2 = torch_utils.time_synchronized()
        # to float
        if self.half:
            pred = pred.float()
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold,
                                   multi_label=False, classes=self.classes, agnostic=self.agnostic)
        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.model_classifier, img, im0s)
        # Process detections
        for i, det in enumerate(pred):  # detections for image i
            
            if self.source.isnumeric():  # batch_size >= 1
                p, s, self.img = path[i], '%g: ' % i, im0s[i].copy() #im0s.copy()
            else:
                p, s, self.img = path, '', im0s
            self.video_path = str(Path(self.outputDIR) / Path(p).name)
            self.outpath = str(Path(self.outputDIR) / "outImg") + "/" + str(j) + ".png" # self.outputDIR + "/outImg/" + str(Path(p).name)
        
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(self.img.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            label = None
            if det is not None and len(det):
                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.img.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %s, ' % (n, self.classes_name[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if self.save_results:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(self.video_path[:self.video_path.rfind('.')] + '.txt', 'a') as file:
                            file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                    if self.show_img or self.view_image:  # Add bbox to image
                        # label = '%s %.2f' % (names[int(cls)], conf)
                        label = '%s' % (self.classes_name[int(cls)])
                        plot_one_box(xyxy, self.img, label=label, color=self.colors[int(cls)])
            if(label == None):
                label = "n"
            print('%sDone. (%.3fs)' % (s, t2 - t1))
            if(self.show_img):
                self.showResults(p , j, label)
            
    def evaluate_dataset(self):
        for path, img, im0s, self.video_cap, img_name, self.nframes in self.dataset:
            
            # img = np.ones((3,320,512), dtype=int)*114
            #im0s = cv2.imread(self.source)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            self.evaluate_frame(im0s, img, img_name, path) #self.source.split("/")[0]
        
def main():
    net = Yolov3()
    net.setConfigs()
    net.setNetwork()

    if(opt.dataset == "True"):
        net.evaluate_dataset()
    else:
        net.evaluate_frame(img)
    if net.save:
        print('Results saved to %s' % os.getcwd() + os.sep + net.outputDIR)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)
        print(net.list_result)
        with open("outputs/test.json", 'w+') as outfile:
            outfile.write(dumps(net.list_result, indent=4))

    print('Done. (%.3fs)' % (time.time() - net.t0))
    #print(f"Average FPS: {net.nframes/(time.time() - net.t0)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='../model/cfg/labels/yolov3-spp-2cls.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='../model/data/traffic_light_2.names', help='*.names path')
    parser.add_argument('--weights', type=str, required=True, help='weights path')
    parser.add_argument('--source', type=str, default='input/videos/t5_5.mp4', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='outputs', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--dataset', type=str, default='True', help='Use image or dataset')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)

    with torch.no_grad():
        main()
