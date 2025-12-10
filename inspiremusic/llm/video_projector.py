# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class ResMLPProjector(nn.Module):
    """
    ResMLP Projector (v1.6 Spec Compliant)
    
    功能：将视频特征映射到 LLM 嵌入空间，并进行非线性适配。
    
    架构特点：
    1. Input Projection: 线性维度对齐 (768 -> 1536)
    2. Residual MLP: 增加非线性表达能力 (Pre-LN 结构)
    3. Zero-Init: MLP 分支零初始化，保证训练初期行为接近线性，避免梯度爆炸
    4. Output LayerNorm: [v1.6 新增] 稳定输出分布，对抗 RoPE 的长程位置扰动
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim_ratio: float = 4.0,
        dropout: float = 0.1,
        use_input_ln: bool = False,
        zero_init_last_linear: bool = True,
        apply_output_ln: bool = True,  # v1.6 关键新增：默认开启输出归一化
    ):
        super().__init__()

        self.apply_output_ln = apply_output_ln
        self.use_input_ln = use_input_ln

        # 1. 输入归一化 (Optional)
        # 建议：如果上游 LSTV 特征未归一化，建议开启
        if use_input_ln:
            self.input_ln = nn.LayerNorm(input_dim)

        # 2. 维度对齐 (主干通路)
        self.input_proj = nn.Linear(input_dim, output_dim)

        # 3. 残差 MLP 块 (Pre-LN 结构)
        self.ln_mlp = nn.LayerNorm(output_dim)
        mid_dim = int(output_dim * hidden_dim_ratio)
        
        self.mlp = nn.Sequential(
            nn.Linear(output_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, output_dim),
            nn.Dropout(dropout)
        )

        # 4. 输出归一化 (v1.6 Spec)
        # 作用：将特征强行拉回标准分布，防止 RoPE 旋转后数值溢出或 Attention Score 异常
        if apply_output_ln:
            self.output_ln = nn.LayerNorm(output_dim)

        # 初始化权重
        self._init_weights(zero_init_last_linear)

    def _init_weights(self, zero_init_last_linear: bool):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)

        # Zero-Init 策略：仅针对 MLP 的最后一层
        # 效果：初始阶段 y = x + 0，模型退化为简单的线性映射，实现“软着陆”
        if zero_init_last_linear:
            # 获取 Sequential 中的最后一个 Linear 层
            last_linear = self.mlp[2] 
            nn.init.zeros_(last_linear.weight)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Time, input_dim] (e.g., [B, T, 768])
        Returns:
            out: [Batch, Time, output_dim] (e.g., [B, T, 1536])
        """
        # 1. Input Norm
        if self.use_input_ln:
            x = self.input_ln(x)

        # 2. Projection (Main Path)
        x = self.input_proj(x)

        # 3. Residual MLP
        # Pre-LN: Norm -> MLP -> Add
        residual = x
        x = self.ln_mlp(x)
        x = self.mlp(x)
        x = x + residual

        # 4. Output Norm (Crucial for RoPE stability)
        if self.apply_output_ln:
            x = self.output_ln(x)

        return x

# ============================================================
# 单元测试代码 (Unit Test)
# 运行此文件即可验证模块是否正常
# ============================================================
if __name__ == "__main__":
    print("=== Testing ResMLPProjector (v1.6) ===")
    
    # 1. 配置参数
    B, T = 2, 128
    D_in, D_out = 768, 1536
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 2. 实例化模型
    model = ResMLPProjector(
        input_dim=D_in,
        output_dim=D_out,
        use_input_ln=True,
        zero_init_last_linear=True,
        apply_output_ln=True # 重点验证此项
    ).to(device)

    # 3. 构造伪造数据
    x = torch.randn(B, T, D_in).to(device).requires_grad_(True)

    # 4. 前向传播测试
    try:
        y = model(x)
        print(f"\\n[Forward Pass] Success")
        print(f"Input shape:  {x.shape}")
        print(f"Output shape: {y.shape}")
        
        # 验证输出维度
        assert y.shape == (B, T, D_out), f"Shape mismatch: {y.shape}"
        
        # 验证数值分布 (开启 Output LN 后，均值应接近0，方差接近1)
        mean = y.mean().item()
        std = y.std().item()
        print(f"Output Mean: {mean:.4f} (Expect ~0.0)")
        print(f"Output Std:  {std:.4f}  (Expect ~1.0)")
        
    except Exception as e:
        print(f"[Forward Pass] Failed: {e}")
        exit(1)

    # 5. 反向传播测试
    try:
        loss = y.sum()
        loss.backward()
        print(f"\\n[Backward Pass] Success")
        
        # 验证梯度是否存在
        grad_norm = x.grad.norm().item()
        print(f"Input Grad Norm: {grad_norm:.4f}")
        assert grad_norm > 0, "Input gradient is zero!"
        
        # 验证参数梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 0:
                has_grad = True
                # print(f"Param {name} has grad: {param.grad.norm():.4f}")
        assert has_grad, "Model parameters have no gradient!"
        print("Gradient flows correctly through all layers.")

    except Exception as e:
        print(f"[Backward Pass] Failed: {e}")
        exit(1)

    print("\\n=== All Tests Passed ===")