import onnx
import numpy as np
from onnx import helper
from onnx import TensorProto

# 入力情報の定義
input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 2000])

# 出力情報の定義
output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 2000])

# Reshapeノードの定義
reshape_node = helper.make_node(
    'Reshape',                      # ノードの種類
    inputs=['input', 'reshape_shape'], # 入力名（2番目はReshapeのターゲット形状）
    outputs=['output']              # 出力名を直接モデルの出力に接続
)

# Reshapeのターゲット形状を指定する定数ノード
reshape_shape = helper.make_tensor(
    name='reshape_shape',           # 定数名
    data_type=TensorProto.INT64,    # データ型
    dims=[2],                       # 定数の形状
    vals=np.array([1, 2000], dtype=np.int64).tolist() # 定数値
)

# グラフの作成
graph = helper.make_graph(
    nodes=[reshape_node],           # Reshapeノードのみ
    name='SimpleReshapeModel',      # モデル名
    inputs=[input_tensor],          # 入力テンソル
    outputs=[output_tensor],        # 出力テンソル
    initializer=[reshape_shape]     # 初期化データ
)

# モデルの作成
model = helper.make_model(graph, producer_name='custom_reshape_model')

# モデルの保存
onnx.save(model, 'reshape_direct.onnx')

print("モデル 'reshape_direct.onnx' を作成し、保存しました。")
