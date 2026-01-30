import numpy as np
import tensorflow as tf

loss_class = classifier_train(model_classifier,
                              [image[valid_roi], x_roi],
                              [y_class_label, y_classifier])

loss_class = [0, 0]
for j in range(len(valid_roi)):
    loss = classifier_train(model_classifier,
                            [np.expand_dims(image[valid_roi[j]], axis=0),
                             np.expand_dims(x_roi[j], axis=0)],
                            [np.expand_dims(y_class_label[j], axis=0),
                             np.expand_dims(y_classifier[j], axis=0)])
    loss_class[0] += loss[0]
    loss_class[1] += loss[1]

loss_class[0] /= len(valid_roi)
loss_class[1] /= len(valid_roi)

loss_class = classifier_train(model_classifier,
                              [np.expand_dims(image[valid_roi[0]], axis=0),
                               np.expand_dims(x_roi[0], axis=0)],
                              [np.expand_dims(y_class_label[0], axis=0),
                               np.expand_dims(y_classifier[0], axis=0)])

def class_loss_regr(num_classes):
    epsilon = 1e-4

    def class_loss_regr_fixed_num(y_true, y_pred):
        """

        :param y_true:  [batch_size, num_rois, num_classes * 8]
                        [:, :, :num_classes * 4] is label index, [:, :, num_classes * 4:] is ground true boxes coordinate.
        :param y_pred: [batch_size, num_rois, num_classes * 4]
        :return: classifier regr_loss
        """
        regr_loss = 0
        batch_size = len(y_true)
        for i in range(batch_size):
            x = y_true[i, :, 4 * num_classes:] - y_pred[i, :, :]                  
            x_abs = backend.abs(x)                                                  
            x_bool = backend.cast(backend.less_equal(x_abs, 1.0), 'float32')       

            loss = 4 * backend.sum(
                y_true[i, :, :4 * num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / backend.sum(
                epsilon + y_true[i, :, :4 * num_classes])
            regr_loss += loss

        return regr_loss / backend.constant(batch_size)

    return class_loss_regr_fixed_num

def main():
    global rpn_optimizer, classifier_optimizer
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    share_layer = ResNet50(img_input)
    rpn = frcnn.rpn(share_layer, num_anchors=len(cfg.anchor_box_ratios) * len(cfg.anchor_box_scales))
    classifier = frcnn.classifier(share_layer, roi_input, cfg.num_rois, nb_classes=cfg.num_classes)

    model_rpn = models.Model(img_input, rpn)
    model_classifier = models.Model([img_input, roi_input], classifier)
    model_all = models.Model([img_input, roi_input], rpn + classifier)

    anchors = get_anchors(cfg.share_layer_shape, cfg.input_shape)

    box_parse = BoundingBox(anchors, max_threshold=cfg.rpn_max_overlap, min_threshold=cfg.rpn_min_overlap)

    reader = DataReader(cfg.annotation_path, box_parse, cfg.batch_size)
    train_data = reader.generate()
    train_step = len(reader.train_lines) // cfg.batch_size

    losses = np.zeros((train_step, 4))
    best_loss = np.Inf

    rpn_lr = CosineAnnealSchedule(cfg.epoch, train_step, cfg.rpn_lr_max, cfg.rpn_lr_min)
    cls_lr = CosineAnnealSchedule(cfg.epoch, train_step, cfg.cls_lr_max, cfg.cls_lr_min)

    rpn_optimizer = optimizers.Adam(rpn_lr)
    classifier_optimizer = optimizers.Adam(cls_lr)

    for e in range(cfg.epoch):
        invalid_data = 0        
        print("Learning rate adjustment, rpn_lr: {}, cls_lr: {}".
              format(rpn_optimizer._decayed_lr("float32").numpy(),
                     classifier_optimizer._decayed_lr("float32").numpy()))


        progbar = utils.Progbar(train_step)
        print('Epoch {}/{}'.format(e+1, cfg.epoch))
        for i in range(train_step):

            image, rpn_y, bbox = next(train_data)
            loss_rpn = rpn_train(model_rpn, image, rpn_y)
            predict_rpn = model_rpn(image)

            predict_boxes = box_parse.detection_out(predict_rpn, confidence_threshold=0)
            height, width = np.shape(image[0])[:2]
            x_roi, y_class_label, y_classifier, valid_roi = get_classifier_train_data(predict_boxes,
                                                                                      bbox,
                                                                                      width,
                                                                                      height,
                                                                                      cfg.batch_size,
                                                                                      cfg.num_classes)

            invalid_data += (cfg.batch_size - len(valid_roi))
            if len(x_roi) == 0:
                progbar.update(i+1, [('rpn_cls', np.mean(losses[:i+1, 0])),
                                     ('rpn_regr', np.mean(losses[:i+1, 1])),
                                     ('detector_cls', np.mean(losses[:i+1, 2])),
                                     ('detector_regr', np.mean(losses[:i+1, 3]))])
                continue
            
            #######you can test code here#########
            loss_class = classifier_train(model_classifier,
                                          [image[valid_roi], x_roi],
                                          [y_class_label, y_classifier])
            #################################

            losses[i, 0] = loss_rpn[0].numpy()
            losses[i, 1] = loss_rpn[1].numpy()
            losses[i, 2] = loss_class[0].numpy()
            losses[i, 3] = loss_class[1].numpy()

            progbar.update(i+1, [('rpn_cls', np.mean(losses[:i+1, 0])),
                                 ('rpn_regr', np.mean(losses[:i+1, 1])),
                                 ('detector_cls', np.mean(losses[:i+1, 2])),
                                 ('detector_regr', np.mean(losses[:i+1, 3]))])

        else:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])

            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr

            print('\nLoss RPN classifier: {:.4f}'.format(loss_rpn_cls))
            print('Loss RPN regression: {:.4f}'.format(loss_rpn_regr))
            print('Loss Detector classifier: {:.4f}'.format(loss_class_cls))
            print('Loss Detector regression: {:.4f}'.format(loss_class_regr))
            print("{} picture can't detect any roi.".format(invalid_data))

            print('The best loss is {:.4f}. The current loss is {:.4f}.'.format(best_loss, curr_loss))
            if curr_loss < best_loss:
                best_loss = curr_loss

            print('Saving weights.\n')
            model_all.save_weights("./logs/model/voc_{:.4f}.h5".format(curr_loss))