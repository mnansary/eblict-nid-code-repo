#-*- coding: utf-8 -*-
"""
@author:Mobassir Hossain,MD.Nazmuddoha Ansary
"""
from __future__ import print_function
#-------------------------
# imports
#-------------------------
import onnxruntime as ort
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import torch
import torchvision
import numpy as np

#-------------------------
# model
#------------------------


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def non_max_suppression(prediction, 
                        conf_thres=0.25, 
                        iou_thres=0.45, 
                        classes=None, 
                        agnostic=False, 
                        multi_label=False,
                        labels=(), 
                        max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    prediction=torch.tensor(prediction)
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        
    return to_numpy(output[0]) 

class YOLO(object):
    def __init__(self,
                model_weights,
                labels, 
                providers=['CUDAExecutionProvider'],
                img_dim=(512,512),
                graph_input="images"):
        self.img_dim=img_dim
        self.classes=[i for i in range(len(labels))]
        self.graph_input=graph_input
        self.model = ort.InferenceSession(model_weights, providers=providers)
        self.labels=labels
    
    def process(self,data):
        h,w,_=data.shape
        data=cv2.resize(data,self.img_dim)
        data=np.transpose(data,(2,0,1))
        data=np.expand_dims(data,axis=0)
        data=data/255
        data=data.astype(np.float32)
        out=self.model.run(None,{"images":data})[0]
        bbox=non_max_suppression(out,classes=self.classes)
        
        res={}
        for box in bbox:
            x1,y1,x2,y2,_,cidx=box
            x1=int(w*(x1/self.img_dim[1]))
            y1=int(h*(y1/self.img_dim[0]))
            x2=int(w*(x2/self.img_dim[1]))
            y2=int(h*(y2/self.img_dim[0]))
            box=[x1,y1,x2,y2]
            res[self.labels[int(cidx)]]=box
        
        for label in self.labels:
            if label not in res.keys():
                res[label]=None
        return res

    def locate_data(self,img,angle):
        if angle!=0:
            data=cv2.rotate(img,angle)
        else:
            data=np.copy(img)
        locs=self.process(data)
        return locs,data

    def draw_rois(self,data,locs):
        img=np.copy(data)
        for k in locs.keys():
            if locs[k] is not None:
                x1,y1,x2,y2=locs[k]
                img=cv2.rectangle(img,(x1,y1),(x2,y2),(255,255,0),thickness=5)
        return img

    def check_ename(self,locs):
        if locs["ename"] is None:
            if locs["bname"] is not None and locs["fname"] is not None:
                x1b,y1b,x2b,y2b=locs["bname"]
                x1f,y1f,x2f,y2f=locs["fname"]
                
                x1e=min(x1b,x1f)
                x2e=max(x2b,x2f)
                ymid=y2b+(y1f-y2b)//2
                h2=(y2b-y1b)//2
                y1e=ymid-h2 
                y2e=ymid+h2
                locs["ename"]=[x1e,y1e,x2e,y2e]
                return locs
            else:
                return locs
        else:
            return locs




    def check_rois(self,clss,locs):
        found=True
        founds=[]
        if "ename" in clss:
            locs=self.check_ename(locs)
            
        for cls in clss:
            if locs[cls] is None:
                found=False
                return found,founds
            else:
                founds.append(cls)
        return found,founds
    
    def align_boxes(self,locs):
        if len(locs.keys())>2:
            max_x1=max([locs[_key][0] for _key in locs.keys() if locs[_key] is not None and _key not in ["nid","dob"]])
            for k in locs.keys():
                if k not in ["nid","dob"] and locs[k] is not None:
                    locs[k][0]=max_x1
            return locs
        else: 
            return locs

    def __call__(self,img,clss,debug=False):
        '''
            args:
                img : the image to process
                clss: the classes to look for
        '''
        angles=[0,cv2.ROTATE_180]
        found=False
        for angle in angles:
            locs,data=self.locate_data(img,angle)
            if debug:
                viz=self.draw_rois(data,locs)
                print(locs)
                plt.imshow(viz)
                plt.show()

            _found,founds=self.check_rois(clss,locs)
            if _found:
                img=np.copy(data)
                found=True
                locs=self.align_boxes(locs)
                break
        
        if found:
            return img,locs,founds
        else:
            return None,None,founds