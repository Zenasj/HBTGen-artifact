import torch.nn as nn

class TestBatchedNMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                num_classes,
                topk,
                keep_topk,
                score_threshold,
                iou_threshould,
                is_normalized=False,
                clip_boxes=False,
                share_location=False,
                background_label_id=-1):
        # Pytorch operations in here
        num_detections = ...
        boxes = ...
        scores = ...
        labels = ...
        return num_detections, boxes, scores, labels

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 num_classes,
                 topk,
                 keep_topk,
                 score_threshold,
                 iou_threshould,
                 is_normalized=False,
                 clip_boxes=False,
                 share_location=False,
                 background_label_id=-1):
        return g.op(
            'BatchedNMS',
            boxes,
            scores,
            numClasses_i=num_classes,
            topK_i=topk,
            keepTopK_i=keep_topk,
            scoreThreshold_f=score_threshold,
            iouThreshold_f=iou_threshould,
            isNormalized_i=is_normalized,
            clipBoxes_i=clip_boxes,
            shareLocation_i=share_location,
            backgroundLabelId_i=background_label_id)

import torch


class TestBatchedNMSop(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                num_classes,
                topk,
                keep_topk,
                score_threshold,
                iou_threshould,
                is_normalized=False,
                clip_boxes=False,
                share_location=False,
                background_label_id=-1):
        
        return TestBatchedNMSop.output

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 num_classes,
                 topk,
                 keep_topk,
                 score_threshold,
                 iou_threshould,
                 is_normalized=False,
                 clip_boxes=False,
                 share_location=False,
                 background_label_id=-1):
        return g.op(
            'BatchedNMS_TRT',
            boxes,
            scores,
            numClasses_i=num_classes,
            topK_i=topk,
            keepTopK_i=keep_topk,
            scoreThreshold_f=score_threshold,
            iouThreshold_f=iou_threshould,
            isNormalized_i=is_normalized,
            clipBoxes_i=clip_boxes,
            shareLocation_i=share_location,
            backgroundLabelId_i=background_label_id,
            outputs=len(TestBatchedNMSop.output))

class TestModel(torch.nn.Module):
    def __init__(self,
                num_classes=80,
                topk=10000,
                keep_topk=1000,
                score_threshold=0.05,
                iou_threshould=0.5,
                is_normalized=False,
                clip_boxes=False,
                share_location=False,
                background_label_id=-1):
        super(TestModel, self).__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.keep_topk = keep_topk
        self.score_threshold = score_threshold
        self.iou_threshould = iou_threshould
        self.is_normalized = is_normalized
        self.clip_boxes = clip_boxes
        self.share_location = share_location
        self.background_label_id = background_label_id
  
    def forward(self, boxes, scores):
        # turn off tracing
        #####################Dose not Work #####################
        state = torch._C._get_tracing_state() # does not work
        #####################Dose not Work #####################
        # do normal process
        batch_size = boxes.shape[0]
        num_detections = torch.zeros(batch_size, 1) + 100
        nmsed_boxes = torch.ones(batch_size, self.keep_topk, 4)
        nmsed_scores = torch.ones(batch_size, self.keep_topk)
        nmsed_classes = torch.ones(batch_size, self.keep_topk)
        # directly save output, so our temporarily
        # Customop does not need to implement any
        # special code
        output = (num_detections, nmsed_boxes, nmsed_scores, nmsed_classes)
        setattr(TestBatchedNMSop, 'output', output)
        # open tracing
        #####################Dose not Work #####################
        torch._C._set_tracing_state(state) # does not work
        #####################Dose not Work #####################
        # we do not need to save the output
        # just call it for creating a correspond
        # node in graph
        return TestBatchedNMSop.apply(boxes, 
                                      scores, 
                                      self.num_classes,
                                      self.topk, 
                                      self.keep_topk, 
                                      self.score_threshold, 
                                      self.iou_threshould,
                                      self.is_normalized, 
                                      self.clip_boxes, 
                                      self.share_location, 
                                      self.background_label_id)

boxes = torch.ones(1, 8732, 80, 4)
scores = torch.ones(1, 8732, 80)

model = TestModel().eval()
torch.onnx.export(
    model, 
    (boxes, scores),
    "batchednms_trt.onnx",
    input_names=['boxes', 'scores'],
    output_names=['num_detections', 'nmsed_boxes', 'nmsed_scores', 'nmsed_classes'],
    export_params=True,
    keep_initializers_as_inputs=True,
    enable_onnx_checker=False,
    verbose=True)