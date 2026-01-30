import torch
import torchvision

def non_max_suppression(prediction: Tensor,
    conf_thres: float=0.25,
    iou_thres: float=0.45000000000000001,
    classes: Tensor=CONSTANTS.c0,
    labels: Tensor=CONSTANTS.c0) -> List[Tensor]:
  _0 = uninitialized(bool)   ## <============== I believe this is the cause of prim_Uninitialized operator.
  nc = torch.sub((torch.size(prediction))[2], 5)
  xc = torch.gt(torch.select(prediction, -1, 4), conf_thres)
  multi_label = torch.gt(nc, 1)
  _1 = ops.prim.device(prediction)
  _2 = torch.zeros([0, 6], dtype=None, layout=None, device=_1, pin_memory=None)
  output = torch.mul([_2], (torch.size(prediction))[0])
  _3 = [9223372036854775807, torch.len(prediction)]
  _4 = ops.prim.min(_3)
  _5 = 0
  _6 = torch.gt(_4, 0)
  while _6:
    x = torch.select(prediction, 0, _5)
    _7 = annotate(List[Optional[Tensor]], [torch.select(xc, 0, _5)])
    x0 = torch.index(x, _7)
    if bool(labels):
      _9 = bool(torch.len(torch.size(labels)))
      _8 = _9
    else:
      _8 = False
    if _8:
      _11 = torch.len(torch.select(labels, 0, _5))
      _10 = bool(_11)
    else:
      _10 = False
    if _10:
      l = torch.select(labels, 0, _5)
      _12 = torch.len(l)
      _13 = torch.add(nc, 5)
      _14 = ops.prim.device(x0)
      v = torch.zeros([_12, _13], dtype=None, layout=None, device=_14, pin_memory=None)
      _15 = torch.slice(l, 0, 0, 9223372036854775807, 1)
      _16 = torch.slice(_15, 1, 1, 5, 1)
      _17 = torch.slice(v, 0, 0, 9223372036854775807, 1)
      _18 = torch.copy_(torch.slice(_17, 1, 0, 4, 1), _16, False)
      _19 = torch.slice(v, 0, 0, 9223372036854775807, 1)
      _20 = torch.select(_19, 1, 4)
      _21 = torch.tensor(1., dtype=ops.prim.dtype(_20), device=ops.prim.device(_20), requires_grad=False)
      _22 = torch.copy_(_20, _21, False)
      _23 = torch.arange(torch.len(l), dtype=None, layout=None, device=None, pin_memory=None)
      _24 = torch.slice(l, 0, 0, 9223372036854775807, 1)
      _25 = torch.to(torch.select(_24, 1, 0), 4, False, False, None)
      _26 = torch.add(_25, 5, 1)
      _27 = torch.tensor(1., dtype=ops.prim.dtype(v), device=ops.prim.device(v), requires_grad=False)
      _28 = annotate(List[Optional[Tensor]], [_23, _26])
      _29 = torch.index_put_(v, _28, _27, False)
      x1 = torch.cat([x0, v], 0)
    else:
      x1 = x0
    _30 = torch.__not__(bool((torch.size(x1))[0]))
    if _30:
      _31, _32, _33 = True, True, _0
    else:
      _34 = torch.slice(x1, 0, 0, 9223372036854775807, 1)
      _35 = torch.slice(_34, 1, 5, 9223372036854775807, 1)
      _36 = torch.slice(x1, 0, 0, 9223372036854775807, 1)
      _37 = torch.mul_(_35, torch.slice(_36, 1, 4, 5, 1))
      _38 = torch.slice(x1, 0, 0, 9223372036854775807, 1)
      box = __torch__.utils.general.xywh2xyxy(torch.slice(_38, 1, 0, 4, 1), )
      if multi_label:
        _39 = torch.slice(x1, 0, 0, 9223372036854775807, 1)
        _40 = torch.slice(_39, 1, 5, 9223372036854775807, 1)
        _41 = torch.nonzero(torch.gt(_40, conf_thres))
        tmp = torch.numpy_T(_41)
        i = torch.select(tmp, 0, 0)
        j = torch.select(tmp, 0, 1)
        _42 = annotate(List[Optional[Tensor]], [i])
        _43 = torch.index(box, _42)
        _44 = torch.add(j, 5, 1)
        _45 = torch.unsqueeze(x1, 2)
        _46 = annotate(List[Optional[Tensor]], [i, _44])
        _47 = torch.index(_45, _46)
        _48 = torch.slice(j, 0, 0, 9223372036854775807, 1)
        _49 = torch.to(torch.unsqueeze(_48, 1), 6, False, False, None)
        x2 = torch.cat([_43, _47, _49], 1)
      else:
        _50 = torch.slice(x1, 0, 0, 9223372036854775807, 1)
        _51 = torch.slice(_50, 1, 5, 9223372036854775807, 1)
        conf, j0 = torch.max(_51, 1, True)
        _52 = torch.to(j0, 6, False, False, None)
        _53 = torch.cat([box, conf, _52], 1)
        _54 = torch.gt(torch.view(conf, [-1]), conf_thres)
        _55 = annotate(List[Optional[Tensor]], [_54])
        x2 = torch.index(_53, _55)
      if bool(classes):
        _57 = bool(torch.len(torch.size(classes)))
        _56 = _57
      else:
        _56 = False
      if _56:
        _58 = torch.slice(x2, 0, 0, 9223372036854775807, 1)
        _59 = torch.slice(_58, 1, 5, 6, 1)
        _60 = ops.prim.device(x2)
        _61 = torch.tensor(annotate(float, classes), dtype=None, device=_60, requires_grad=False)
        _62 = torch.any(torch.eq(_59, _61), 1, False)
        _63 = annotate(List[Optional[Tensor]], [_62])
        x3 = torch.index(x2, _63)
      else:
        x3 = x2
      n = (torch.size(x3))[0]
      if torch.__not__(bool(n)):
        _64, _65, _66 = True, True, _0
      else:
        _67 = torch.slice(x3, 0, 0, 9223372036854775807, 1)
        c = torch.mul(torch.slice(_67, 1, 5, 6, 1), 4096)
        _68 = torch.slice(x3, 0, 0, 9223372036854775807, 1)
        boxes = torch.add(torch.slice(_68, 1, 0, 4, 1), c, alpha=1)
        _69 = torch.slice(x3, 0, 0, 9223372036854775807, 1)
        scores = torch.select(_69, 1, 4)
        i0 = __torch__.torchvision.ops.boxes.nms(boxes, scores, iou_thres, )
        _70 = torch.gt((torch.size(i0))[0], 300)
        if _70:
          i1 = torch.slice(i0, 0, 0, 300, 1)
        else:
          i1 = i0
        _71 = annotate(List[Optional[Tensor]], [i1])
        _72 = torch._set_item(output, _5, torch.index(x3, _71))
        _64, _65, _66 = False, _0, True
      _31, _32, _33 = _64, _65, _66
    if _31:
      _73 = _32
    else:
      _73 = _33
    _74 = torch.add(_5, 1)
    _75 = torch.__and__(torch.lt(_74, _4), _73)
    _6, _5 = _75, _74
  return output