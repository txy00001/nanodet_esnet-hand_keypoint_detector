import onnx_tool
import torch
from onnx_tool import model_profile

model_path1 = 'onnx/out.onnx'
#model_path2 = './model/nanodet-plus-m_416.onnx'

onnx_tool.model_profile(model_path1, savenode='onnx/out.onnx.txt')


from onnx_opcounter import calculate_params
import onnx

model = onnx.load_model('onnx/out.onnx')
params = calculate_params(model)

print('Number of params:', params)


#
# from thop import profile
# # input = torch.randn(1, 3, 416, 416)
# macs, params = profile(model, inputs=(input, ))
# print('Total macc:{}, Total params: {}'.format(macs, params))