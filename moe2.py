# moe_with_existing_expert.py
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

from modeling import CANTRIPModel


# ==================================================================
# 1. 定义现有神经网络专家（替换为你的模型）
# ==================================================================
def build_existing_expert(input_shape, output_shape, **kwargs):
    """
    将CANTRIP模型封装为MoE专家
    参数:
        input_shape: 输入数据的形状（需与CANTRIP输入匹配）
        output_shape: 输出维度（需与num_classes一致）
        **kwargs: CANTRIP模型的其他参数（需与原始模型一致）
    """
    # 确保输出维度匹配
    assert output_shape == kwargs.get('num_classes', 2), "输出维度不匹配"

    # 创建CANTRIP模型实例
    expert = CANTRIPModel(
        max_seq_len=input_shape[1],  # 假设输入形状为(batch_size, seq_len, features)
        max_snapshot_size=input_shape[2],  # 根据实际数据调整
        vocabulary_size=kwargs['vocab_size'],
        observation_embedding_size=128,  # 根据实际配置调整
        delta_encoding_size=1,  # 根据实际配置调整
        num_hidden=256,  # 根据实际配置调整
        cell_type='LRAN',  # 根据实际配置调整
        batch_size=None,  # 动态batch_size
        snapshot_encoder=kwargs['encoder'],  # 传入你的编码函数
        dropout=0.3,  # 根据实际配置调整
        num_classes=output_shape,
        **kwargs  # 其他参数
    )

    # 创建输入层（动态batch_size）
    inputs = tf.keras.Input(shape=input_shape[1:], batch_size=None)

    # 调用模型（需要实现call方法）
    outputs = expert(inputs)

    # 封装为Keras模型
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


# ==================================================================
# 2. 构建MoE模型
# ==================================================================
def build_moe_model(input_shape, num_classes, num_experts=3, use_shared_features=False):
    """
    构建MoE模型
    参数:
        input_shape: 输入数据的形状
        num_classes: 输出类别数
        num_experts: 专家网络数量
        use_shared_features: 是否使用共享特征层
    """
    inputs = layers.Input(shape=input_shape)

    # 可选：共享特征提取层
    if use_shared_features:
        shared_features = layers.Dense(128, activation='relu')(inputs)
    else:
        shared_features = inputs

    # 初始化专家网络
    experts = []
    # 将现有模型作为第一个专家（可冻结权重）
    expert1 = build_existing_expert(input_shape if not use_shared_features else 128, num_classes)
    expert1.trainable = True  # 可选：是否冻结现有模型
    experts.append(expert1)

    # 添加新专家（可选：使用不同结构）
    for _ in range(num_experts - 1):
        new_expert = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape if not use_shared_features else 128),
            layers.Dense(32, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
        ])
        experts.append(new_expert)

    # 门控网络
    gate = layers.Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(num_experts, activation='softmax')
    ])(shared_features)

    # MoE聚合
    outputs = []
    for i in range(num_experts):
        expert_output = experts[i](shared_features)
        outputs.append(expert_output * gate[:, i:i + 1])  # 广播机制

    combined_output = layers.Add()(outputs)

    # 构建完整模型
    model = Model(inputs=inputs, outputs=combined_output)

    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# ==================================================================
# 3. 示例数据和训练流程
# ==================================================================
def main():
    # 示例数据（替换为你的实际数据）
    num_samples = 1000
    input_shape = (50,)  # 根据你的数据调整
    num_classes = 3  # 根据你的任务调整

    # 生成虚拟数据
    X = np.random.rand(num_samples, input_shape[0])
    y = np.random.randint(0, num_classes, num_samples)
    y_onehot = tf.keras.utils.to_categorical(y, num_classes=num_classes)

    # 构建MoE模型
    model = build_moe_model(
        input_shape=input_shape,
        num_classes=num_classes,
        num_experts=3,
        use_shared_features=True
    )

    # 打印模型结构
    model.summary()

    # 训练模型
    history = model.fit(
        X, y_onehot,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )

    # 评估模型
    loss, accuracy = model.evaluate(X, y_onehot, verbose=0)
    print(f"\nFinal Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()