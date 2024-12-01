import onnx
from onnx import helper
from onnx import TensorProto

# モデルの定義
def create_depth_to_space_model():
    # 入力ノード
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2000])

    # 出力ノード
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2000])

    # DepthToSpaceノードを定義
    # blocksizeは DepthToSpace 演算のパラメータ
    depth_to_space_node = helper.make_node(
        "DepthToSpace", 
        inputs=["input"], 
        outputs=["output"], 
        blocksize=2  # 必要に応じて変更
    )

    # # 最終出力ノード
    # identity_node = helper.make_node(
    #     "Identity",
    #     inputs=["depth_to_space_output"],
    #     outputs=["output"]
    # )

    # グラフを作成
    graph = helper.make_graph(
        nodes=[depth_to_space_node],
        name="DepthToSpaceModel",
        inputs=[input_tensor],
        outputs=[output_tensor],
    )

    # モデルを作成
    model = helper.make_model(graph, producer_name="onnx-depth-to-space")

    return model


# モデルを作成して保存
model = create_depth_to_space_model()
onnx.save(model, "DepthToSpace.onnx")
print("モデルが 'DepthToSpace.onnx' に保存されました。")
