import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score


class DiseaseExpert(tf.keras.layers.Layer):
    def __init__(self, expert_units=256, num_experts=3, **kwargs):
        super().__init__(**kwargs)
        # 共享底层特征提取（可选）
        self.shared_layer = tf.keras.layers.Dense(512, activation='relu')
        # 独立专家分支
        self.experts = [
            tf.keras.Sequential([
                tf.keras.layers.Dense(expert_units, activation='relu'),
                tf.keras.layers.Dropout(0.3)
            ]) for _ in range(num_experts)
        ]

    def call(self, inputs):
        shared = self.shared_layer(inputs)
        return [expert(shared) for expert in self.experts]


class GatingNetwork(tf.keras.layers.Layer):
    def __init__(self, num_experts=3, **kwargs):
        super().__init__(**kwargs)
        self.gate_layer = tf.keras.layers.Dense(
            num_experts,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )

    def call(self, inputs):
        return self.gate_layer(inputs)  # 输出权重分布 [batch, num_experts]


class MoEModel(tf.keras.Model):
    def __init__(self, num_classes=3, **kwargs):
        super().__init__(**kwargs)
        self.feature_extractor = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(1024, activation='relu')
        ])
        self.experts = DiseaseExpert(num_experts=3)
        self.gate = GatingNetwork(num_experts=3)
        # 每个病种的独立输出头
        self.task_heads = [
            tf.keras.layers.Dense(1, activation='sigmoid')
            for _ in range(num_classes)
        ]

    def call(self, inputs):
        # 特征提取
        x = self.feature_extractor(inputs)
        # 专家输出
        expert_outputs = self.experts(x)  # [expert1_out, expert2_out, expert3_out]
        # 门控权重
        gate_weights = self.gate(x)  # [batch, 3]

        # 加权组合
        combined = tf.zeros_like(expert_outputs[0])
        for i in range(3):
            weighted = tf.multiply(
                expert_outputs[i],
                tf.expand_dims(gate_weights[:, i], axis=-1)
            )
            combined += weighted

        # 多任务输出
        outputs = [head(combined) for head in self.task_heads]
        return tf.concat(outputs, axis=-1)

    def moe_loss(y_true, y_pred, self=None):
        # y_true shape: [batch, 3]
        # y_pred shape: [batch, 3]
        # 基础交叉熵损失
        ce_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y_true, y_pred)
        )
        # 专家负载均衡正则化（防止专家退化）
        gate_activations = tf.reduce_mean(self.gate.gate_layer.output, axis=0)
        balance_loss = tf.reduce_sum(gate_activations * tf.math.log(gate_activations + 1e-8))
        # 总损失
        return ce_loss + 0.1 * balance_loss


def get_eval_metrics(model, test_data):
    preds = model.predict(test_data)
    metrics = {}
    for i in range(3):
        auc = roc_auc_score(test_labels[:,i], preds[:,i])
        ap = average_precision_score(test_labels[:,i], preds[:,i])
        metrics[f'disease{i+1}_auc'] = auc
        metrics[f'disease{i+1}_ap'] = ap
    return metrics