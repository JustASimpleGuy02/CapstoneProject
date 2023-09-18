# %%
import onnx
import onnxruntime as ort

onnx_model = onnx.load("../onnx/model.onnx")

# %%
output = [node.name for node in onnx_model.graph.output]

input_all = [node.name for node in onnx_model.graph.input]
input_initializer =  [node.name for node in onnx_model.graph.initializer]
net_feed_input = list(set(input_all)  - set(input_initializer))

print('Inputs: ', net_feed_input)
print('Outputs: ', output)

# %%
