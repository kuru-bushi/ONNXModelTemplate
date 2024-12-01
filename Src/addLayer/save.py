import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper

# 入力の定義
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2000])

# 出力の定義
output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2000])

# 定数テンソルの作成 (addレイヤで使用)
add_constant_tensor = helper.make_tensor(
    name="add_constant",
    data_type=onnx.TensorProto.FLOAT,
    dims=[1, 2000],
    vals=[1.0] * 2000  # 全ての値が1の定数
)

# addノードの作成
add_node = helper.make_node(
    "Add",  # 演算タイプ
    ["input", "add_constant"],  # 入力
    ["output"],  # 出力
    name="add_node"
)

# # 出力ノードの接続
# identity_node = helper.make_node(
#     "Identity",  # 入力から出力へ直接コピー
#     ["add_output"],
#     ["output"],
#     name="output_node"
# )

# グラフの定義
graph = helper.make_graph(
    nodes=[add_node],
    name="SimpleAddGraph",
    inputs=[input_tensor],
    outputs=[output_tensor],
    initializer=[add_constant_tensor]
)

# モデルの定義
model = helper.make_model(graph, producer_name="onnx-add-model")

# モデルの保存
onnx.save(model, "add.onnx")

print("Model 'add.onnx' has been saved successfully.")
