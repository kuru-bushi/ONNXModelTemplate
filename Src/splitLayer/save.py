import onnx
import onnx.helper as helper
import onnx.checker as checker

# モデルの構築
# 1. モデルの入力を定義
input_tensor = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2000])

# 2. Split ノードの出力（2つに分割すると仮定）
split_outputs = ["split_output_1", "split_output_2"]

# 3. Split ノード
split_node = helper.make_node(
    "Split",
    inputs=["input"],  # 入力データ
    outputs=["output"],  # 出力名
    axis=1  # 分割する軸
)

# 5. モデルの出力を定義
output_tensor = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2000])

# グラフを作成
graph_def = helper.make_graph(
    nodes=[split_node],
    name="SplitModel",
    inputs=[input_tensor],
    outputs=[output_tensor]
)

# モデルを作成
model_def = helper.make_model(graph_def, producer_name="custom_onnx_model")

# モデルの検証
checker.check_model(model_def)

# モデルを保存
onnx.save(model_def, "split.onnx")

print("Model saved as split.onnx")
