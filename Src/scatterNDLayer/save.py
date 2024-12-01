import numpy as np
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
from onnx import TensorProto

# モデルの作成
def create_scatternd_model():
    # 入力テンソルの定義
    input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2000])
    output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2000])

    # ScatterNDのインデックスと更新値を定義
    indices_shape = [1, 1]  # 単純なインデックス形状
    indices_data = np.array([[100]], dtype=np.int64)  # 例としてインデックス100を指定
    indices_tensor = helper.make_tensor('indices', TensorProto.INT64, indices_shape, indices_data.flatten())

    updates_shape = [1]
    updates_data = np.array([1.0], dtype=np.float32)  # 更新値の例
    updates_tensor = helper.make_tensor('updates', TensorProto.FLOAT, updates_shape, updates_data.flatten())

    # ScatterNDノードを作成
    scatternd_node = helper.make_node(
        'ScatterND',
        inputs=['input', 'indices', 'updates'],
        outputs=['output']
    )

    # グラフを作成
    graph_def = helper.make_graph(
        nodes=[scatternd_node],
        name='ScatterNDGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[indices_tensor, updates_tensor]
    )

    # モデルを作成
    model_def = helper.make_model(graph_def, producer_name='onnx-scatternd-example')
    return model_def

# モデルの生成と保存
model = create_scatternd_model()
onnx.save(model, 'ScatterND.onnx')

print("ScatterND.onnx を作成しました。")
