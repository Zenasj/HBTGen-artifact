import torch
import torch.nn as nn

def create_qconfig_mapping(per_channel_config=None, per_tensor_config=None):

    if per_channel_config is None:
        per_channel_config = default_per_channel_symmetric_qnnpack_qconfig
    if per_tensor_config is None:
        per_tensor_config = default_symmetric_qnnpack_qconfig

    PER_TENSOR_ONLY = [  # take otuside of function later
        torch.nn.functional.linear,
        torch.nn.modules.linear.Linear
    ]

    qconfig_mapping = get_default_qconfig_mapping("qnnpack") \
        .set_global(per_channel_config)

    for pattern in qconfig_mapping.object_type_qconfigs.keys():
        if pattern not in _FIXED_QPARAMS_OP_TO_OBSERVER:
            qconfig_mapping.set_object_type(pattern, per_channel_config)
            if pattern in PER_TENSOR_ONLY:
                qconfig_mapping.set_object_type(pattern, per_tensor_config)

    return qconfig_mapping

def quantize(model,
             device_list,
             trainloader,
             valloader=None,
             epochs=10,
             optimizer=None,
             loss_fn=None,
             scheduler=None,
             val_epoch=None,
             save_dir=None,
             timing=False,
             train_func=None,
             val_func=None,
             logger=None
             ):

    inp = torch.rand((1, 3, 224, 224))

    model_prepared, qm = quant_prepare(model, inp)

    print(qm.to_dict())

    model_calibrated = train_func(
        model=model_prepared,
        trainloader=trainloader,
        valloader=valloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        val_epoch=val_epoch,
        devices=device_list,
        save_dir=save_dir,
        timing=timing,
        val_func=val_func,
        logger=logger
    )

    model_quantized = quant_convert(model_calibrated)

    return model_quantized 

def quant_prepare(model, inputs, qconfig_mapping=None, backend_config=None, engine=None, mode=None):

    if engine is None:
        engine = "qnnpack"
    torch.backends.quantized.engine = engine

    if backend_config is None:
        backend_config = get_qnnpack_backend_config()

    if qconfig_mapping is None:
        qconfig_mapping = create_qconfig_mapping()

    if mode == 'ptq':
        model_prepared = quantize_fx.prepare_fx(
            model,
            qconfig_mapping,
            inputs,
            backend_config=backend_config
        )

    elif mode == 'qat' or mode is None:
        model_prepared = quantize_fx.prepare_qat_fx(
            model,
            qconfig_mapping,
            inputs,
            backend_config=backend_config
        )

    return model_prepared, qconfig_mapping


def quant_convert(model_calibrated, backend_config=None):

    if backend_config is None:
        backend_config = get_qnnpack_backend_config()

    model_calibrated.eval()
    model_quantized = quantize_fx.convert_fx(
        model_calibrated,
        backend_config=backend_config
    )
    return model_quantized

conf_per_channel = QConfig(
        activation=Observers.MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            eps=0.000244140625

        ),
        weight=Observers.MovingAveragePerChannelMinMaxObserver.with_args(
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric,
            reduce_range=False,
            ch_axis=0,
            eps=0.000244140625
        ))

conf_per_tensor = QConfig(
        activation=Observers.MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            eps=0.000244140625

        ),
        weight = Observers.MovingAverageMinMaxObserver.with_args(
            qscheme=torch.per_tensor_symmetric,
            quant_min=-127,
            quant_max=127,
            dtype=torch.qint8,
            eps=0.000244140625

        )
        )