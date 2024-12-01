import onnx
import numpy as np
from onnx import helper, TensorProto

# 入力ノードを作成
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2000])

# 出力ノードを作成
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2000])

# Padノードの属性
pad_values = [0, 0, 1, 1]  # 前後に1ずつパディング
pads = helper.make_tensor('pads', TensorProto.INT64, [4], np.array(pad_values, dtype=np.int64))

# Padノードを作成
pad_node = helper.make_node(
    'Pad',
    inputs=['input', 'pads'],  # 入力データとパディングの情報を渡す
    outputs=['output'],
    mode='constant',
    value=0.0  # パディング値を0とする
)

# # Padノードと出力を接続するためのアイデンティティノードを追加
# identity_node = helper.make_node(
#     'Identity',
#     inputs=['pad_output'],
#     outputs=['output']
# )

# グラフを構築
graph = helper.make_graph(
    # nodes=[pad_node, identity_node],
    nodes=[pad_node],
    name='PadGraph',
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[pads]  # パディングの初期値
)

# モデルを構築
model = helper.make_model(graph, producer_name='pad_model_creator')

# モデルを保存
onnx.save(model, 'pad.onnx')

print("モデルが 'pad.onnx' として保存されました。")
