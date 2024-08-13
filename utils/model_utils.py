def init_qat(model_type:str):
    if model_type == 'trt_torch_online_quantization':
        from pytorch_quantization import nn as quant_nn
        from pytorch_quantization import calib
        from pytorch_quantization.tensor_quant import QuantDescriptor
        from pytorch_quantization import quant_modules
        quant_modules.initialize()