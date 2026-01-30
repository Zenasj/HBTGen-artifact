import tensorflow as tf

def call(self, inputs, training=True):
        if self.mode == "training":
            input_image, input_image_meta, input_gt_class_ids, input_gt_boxes, input_gt_rboxes, input_gt_global_mask, input_gt_masks, input_gt_masks_score,input_rpn_match, input_rpn_bbox = inputs
        else:
            # Anchors in normalized coordinates
            input_image, input_image_meta = inputs
        gt_boxes = norm_boxes_graph(input_gt_boxes, tf.shape(input_image)[1:3])
        gt_rboxes = norm_rboxes_graph(input_gt_rboxes, tf.shape(input_image)[1:3])
        if self.config.BACKBONE == "resnet101":
            C2, C3, C4, C5 = self.backbone(input_image)
        P2, P3, P4, P5, P6 = self.fpn([C2, C3, C4, C5])
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        rpn_class_logits, rpn_class, rpn_bbox = self.rpn(rpn_feature_maps)
        anchors = build_anchors(self.config, image_shape=self.config.IMAGE_SHAPE, norm=True, to_tensor=True)
        rpn_rois = self.proposal([rpn_class, rpn_bbox, anchors])
        if self.mode == "training":
            active_class_ids = parse_image_meta_graph(input_image_meta, self.config)["active_class_ids"]
            target_rois = rpn_rois if self.config.USE_RPN_ROIS else norm_boxes_graph(input_rois, input_image.shape.as_list()[1:3])
            output_rois, target_class_ids, target_bbox, \
                target_embed_length, target_rbox = self.detect_target([target_rois, input_gt_class_ids, gt_boxes, 
                    input_gt_masks, input_gt_rboxes])
            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_box_outputs = self.classifier([output_rois, input_image_meta, mrcnn_feature_maps], True)
            mrcnn_mask = self.masker([output_rois, input_image_meta, mrcnn_feature_maps], True)
            
            return rpn_class_logits, rpn_bbox, mrcnn_box_outputs, mrcnn_mask, target_class_ids, target_bbox, target_mask, active_class_ids

        else:
            mrcnn_box_outputs = self.classifier([rpn_rois, input_image_meta, mrcnn_feature_maps])
            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in
            # normalized coordinates
            detections = self.detect(
                [rpn_rois, mrcnn_box_outputs["mrcnn_class_logits"], 
                mrcnn_box_outputs["mrcnn_bbox"], input_image_meta])
            mrcnn_mask = self.masker([detections[..., :4], input_image_meta, mrcnn_feature_maps])
            return detections, mrcnn_mask

class Loss_tower():
    def __init__(self, config):
        self.config = config

    def rpn_cla_loss(self, rpn_match, rpn_class_logits):
        """RPN anchor classifier loss.
        Params:
        -----------------------------------------------------------
            rpn_match:        [batch, anchors, 1]. Anchor match type. 1=positive,
                            -1=negative, 0=neutral anchor.
            rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
        """
        # Squeeze last dim to simplify
        rpn_match = tf.squeeze(rpn_match, -1)
        # Get anchor classes. Convert the -1/+1 match to 0/1 values.
        anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
        # Positive and Negative anchors contribute to the loss,
        # but neutral anchors (match value = 0) don't.
        indices = tf.where(tf.not_equal(rpn_match, 0))
        # Pick rows that contribute to the loss and filter out the rest.
        rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
        anchor_class = tf.gather_nd(anchor_class, indices)
        if self.config.RPN_CLASS_LOSS_TYPE == 'cross_entropy':
            # Cross entropy loss -------------------------------------------
            loss = tf.losses.sparse_softmax_cross_entropy(labels=anchor_class,
                                                          logits=rpn_class_logits)
            # --------------------------------------------------------------
        elif self.config.RPN_CLASS_LOSS_TYPE == 'focal_loss':
            # Focal loss ---------------------------------------------------
            loss = focal_loss(prediction_tensor=rpn_class_logits, 
                            target_tensor=tf.cast(tf.one_hot(anchor_class, 2), tf.float32))
            # --------------------------------------------------------------
        loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
        # loss = tf.cond(tf.size(loss) > 0, lambda:tf.reduce_mean(loss), lambda:tf.constant(0.0))
        
        return loss
    
    @staticmethod
    def batch_pack_graph(x, counts, num_rows):
        """Picks different number of values from each row
        in x depending on the values in counts.
        """
        outputs = []
        for i in range(num_rows):
            outputs.append(x[i, :counts[i]])
        return tf.concat(outputs, axis=0)

    def rpn_bbox_loss(self, target_bbox, rpn_match, rpn_bbox):
        """Return the RPN bounding box loss graph.
        Params:
        -----------------------------------------------------------
            config:      the model config object.
            target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
                        Uses 0 padding to fill in unsed bbox deltas.
            rpn_match:   [batch, anchors, 1]. Anchor match type. 1=positive,
                        -1=negative, 0=neutral anchor.
            rpn_bbox:    [batch, anchors, (dy, dx, log(dh), log(dw))]
        """
        # Positive anchors contribute to the loss, but negative and
        # neutral anchors (match value of 0 or -1) don't.
        rpn_match = tf.squeeze(rpn_match, -1)
        indices = tf.where(tf.equal(rpn_match, 1))

        # Pick bbox deltas that contribute to the loss
        rpn_bbox = tf.gather_nd(rpn_bbox, indices)

        # Trim target bounding box deltas to the same length as rpn_bbox.
        batch_counts = tf.reduce_sum(tf.cast(tf.equal(rpn_match, 1), tf.int32), axis=1)
        target_bbox = self.batch_pack_graph(target_bbox, batch_counts, self.config.IMAGES_PER_GPU)
        loss = smooth_l1_loss(y_true=target_bbox, y_pred=rpn_bbox, config=self.config)
        loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)
        # loss = tf.cond(tf.size(loss) > 0, lambda:tf.reduce_mean(loss), lambda:tf.constant(0.0))
        return loss

    def mrcnn_cla_loss(self, target_class_ids, pred_class_logits, active_class_ids):
        """Loss for the classifier head of Mask RCNN.
        Params:
        -----------------------------------------------------------
            target_class_ids:  [batch, num_rois]. Integer class IDs. Uses zero
                               padding to fill in the array.
            pred_class_logits: [batch, num_rois, num_classes]
            active_class_ids:  [batch, num_classes]. Has a value of 1 for
                               classes that are in the dataset of the image, and 0
                               for classes that are not in the dataset.
        """
        # During model building, Keras calls this function with
        # target_class_ids of type float32. Unclear why. Cast it
        # to int to get around it.
        target_class_ids = tf.cast(target_class_ids, 'int64')

        # Find predictions of classes that are not in the dataset.
        pred_class_ids = tf.argmax(pred_class_logits, axis=2)
        # TODO: Update this line to work with batch > 1. Right now it assumes all
        #       images in a batch have the same active_class_ids
        pred_active = tf.gather(active_class_ids[0], pred_class_ids)

        if self.config.RCNN_CLASS_LOSS_TYPE == 'cross_entropy':
            # CE_Loss --------------------------------------------------
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_class_ids, logits=pred_class_logits)
            # ----------------------------------------------------------
        elif self.config.RCNN_CLASS_LOSS_TYPE == 'focal_loss':
            # Focal_Loss -----------------------------------------------
            loss = focal_loss(prediction_tensor=pred_class_logits, 
                            target_tensor=tf.cast(tf.one_hot(target_class_ids, \
                                tf.cast(active_class_ids[1][0], "int32")), "float32"))
            loss = tf.reduce_sum(loss, axis=-1)
            # ----------------------------------------------------------

        # Erase losses of predictions of classes that are not in the active
        # classes of the image.
        loss = loss * pred_active

        # Computer loss mean. Use only predictions that contribute
        # to the loss to get a correct mean.
        loss = tf.reduce_sum(loss) / tf.reduce_sum(pred_active)
        return loss


    def mrcnn_bbox_loss(self, target_bbox, target_class_ids, pred_bbox):
        """Loss for Mask R-CNN bounding box refinement.
        Params:
        -----------------------------------------------------------
            target_bbox:      [batch, num_rois, (dy, dx, log(dh), log(dw))]
            target_class_ids: [batch, num_rois]. Integer class IDs.
            pred_bbox:        [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        """
        # Reshape to merge batch and roi dimensions for simplicity.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        target_bbox = tf.reshape(target_bbox, (-1, 4))
        pred_bbox = tf.reshape(pred_bbox, (-1, pred_bbox.shape.as_list()[2], 4))

        # Only positive ROIs contribute to the loss. And only
        # the right class_id of each ROI. Get their indices.
        positive_roi_ix = tf.where(target_class_ids > 0)[:, 0]
        positive_roi_class_ids = tf.cast(tf.gather(target_class_ids, positive_roi_ix), tf.int64)
        indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

        # Gather the deltas (predicted and true) that contribute to loss
        target_bbox = tf.gather(target_bbox, positive_roi_ix)
        pred_bbox = tf.gather_nd(pred_bbox, indices)

        # Smooth-L1 Loss
        if tf.size(target_bbox) > 0:
            loss = smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox, config=self.config)  
        else:
            loss = tf.constant(0.0)
        # loss = tf.cond(tf.size(target_bbox) > 0, 
        #                lambda:smooth_l1_loss(y_true=target_bbox, y_pred=pred_bbox, config=self.config), 
        #                lambda:tf.constant(0.0))
        loss = tf.reduce_mean(loss)
        
        return loss

    def mrcnn_mask_loss(self, target_masks, target_class_ids, pred_masks):
        """Mask binary cross-entropy loss for the masks head.
        Params:
        -----------------------------------------------------------
            target_masks:     [batch, num_rois, height, width].
                            A float32 tensor of values 0 or 1. Uses zero padding to fill array.
            target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
            pred_masks:       [batch, proposals, height, width, num_classes] float32 tensor
                            with values from 0 to 1.
        """
        # Reshape for simplicity. Merge first two dimensions into one.
        target_class_ids = tf.reshape(target_class_ids, (-1,))
        mask_shape = tf.shape(target_masks)
        target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
        pred_shape = tf.shape(pred_masks)
        pred_masks = tf.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3], pred_shape[4]))

        if self.config.MASK_LOSS_TYPE == "binary_ce":
            # NOTE for Compute binary cross entropy ---------------------
            # Permute predicted masks to [N, num_classes, height, width]
            pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

            # Only positive ROIs contribute to the loss. And only
            # the class specific mask of each ROI.
            positive_ix = tf.where(target_class_ids > 0)[:, 0]
            positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int64)
            indices = tf.stack([positive_ix, positive_class_ids], axis=1)

            # Gather the masks (predicted and true) that contribute to loss
            y_true = tf.gather(target_masks, positive_ix)
            # Compute binary cross entropy. 
            # If no positive ROIs, then return 0.
            y_pred = tf.gather_nd(pred_masks, indices)
            if tf.size(y_true) > 0:
                loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
            else:
                loss = tf.constant(0.0)
            # loss = tf.cond(tf.size(y_true) > 0, 
            #             lambda:tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), 
            #             lambda:tf.constant(0.0))
            # ----------------------------------------------------------
        elif self.config.MASK_LOSS_TYPE == "focal_loss":
            # Focal Loss -----------------------------------------------
            # Only positive ROIs contribute to the loss. And only
            # the class specific mask of each ROI.
            positive_ix = tf.where(target_class_ids > 0)[:, 0]
            # Gather the masks (predicted and true) that contribute to loss
            y_true = tf.gather(target_masks, positive_ix)
            y_pred = tf.gather(pred_masks, positive_ix)
            if tf.size(y_true) > 0:
                loss = focal_loss(y_pred, tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.int32), self.config.NUM_CLASSES), 
                                    dtype=tf.float32))
            else:
                loss = tf.constant(0.0)
            # loss = tf.cond(tf.size(y_true) > 0, 
            #             lambda:focal_loss(y_pred, tf.cast(tf.one_hot(tf.cast(y_true, dtype=tf.int32), self.config.NUM_CLASSES), 
            #                         dtype=tf.float32)), 
            #             lambda:tf.constant(0.0))
        
        loss = tf.reduce_mean(loss)

        return loss