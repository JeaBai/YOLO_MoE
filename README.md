<div align="center">
  <h1 style="border-bottom: none; margin-bottom: 0.25rem;">YOLO-MoE: Lightweight Mixture-of-Experts for YOLOv26</h1>
  <p style="margin-top: 0.5rem;">
    <a href="https://www.gnu.org/licenses/agpl-3.0">
      <img src="https://img.shields.io/badge/License-AGPL%203.0-blue.svg" alt="License">
    </a>
    <a href="https://www.python.org/downloads/">
      <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
    </a>
  </p>
  <blockquote style="background-color: #f9f2f4; border-left: 6px solid #d0b0b0; padding: 0.75rem 1rem; margin: 1.5rem 0; border-radius: 4px; color: #5e3a3a;">
    <p style="margin: 0;">
      <strong>实验性项目</strong>：本项目将混合专家（Mixture-of-Experts, MoE）机制引入 YOLOv26 目标检测模型，探索条件计算在边缘检测场景下的可行性。MoE
      核心模块继承自
      <a href="https://github.com/isLinXu/YOLO-Master">YOLO-Master</a>
      ，并针对 YOLOv26 进行了适配与功能增强。目前仍处于实验阶段，存在负载不均衡、数值不稳定等问题，后续将逐步改进。
    </p>
  </blockquote>
</div>

<hr style="margin: 2rem 0; border: 0; border-top: 1px solid #eaecef;">

<!-- 主要工作区块 -->
<div style="margin-bottom: 2.5rem;">
  <h2 style="border-bottom: 1px solid #eaecef; padding-bottom: 0.3rem; margin-top: 1.5rem;">✨ 主要工作</h2>
  <p>
    YOLO-Master 提出了一种高效的“共享专家 + 稀疏专家”双路 MoE 架构，在目标检测中取得了优秀的速度-精度权衡。本项目将该架构集成至
    YOLOv26 框架，并在此基础上进行了以下
    <strong>针对性改进</strong>：
  </p>

  <h3 style="margin-top: 1.8rem; font-weight: 600;">1. 可学习融合门控 (Learnable Fusion Gate)</h3>
  <p>
    官方实现中，共享专家输出与稀疏专家输出通过
    <strong>简单相加</strong>
    进行融合：
  </p>
  <pre style="background-color: #f6f8fa; border-radius: 6px; padding: 16px; overflow: auto; line-height: 1.45;"><code style="background: none; font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;">output = shared_output + expert_output</code></pre>
  <p>
    我们在
    <code style="background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px;">SparseDualMoE</code>
    模块中引入了一个轻量级的可学习门控网络，动态学习两路特征的最佳融合比例：
  </p>
  <pre style="background-color: #f6f8fa; border-radius: 6px; padding: 16px; overflow: auto; line-height: 1.45;"><code style="background: none; font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;">gate_weights = self.fusion_gate(shared_out + expert_out)  # [B, 2, 1, 1]
output = gate_weights[:, 0:1] * shared_out + gate_weights[:, 1:2] * expert_out</code></pre>
  <p>
    这使得模型能够根据输入内容自适应地调节对共享特征与稀疏专家特征的依赖程度。
  </p>

  <h3 style="margin-top: 1.8rem; font-weight: 600;">2. 动态容量控制 (Dynamic Capacity Control)</h3>
  <p>
    我们增加了一个极轻量的复杂度评估器，根据输入特征图动态估计场景复杂度，并据此在线调整激活的专家数量（
    <code style="background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px;">top_k</code>
    ），实现计算资源的“按需分配”：
  </p>
  <pre style="background-color: #f6f8fa; border-radius: 6px; padding: 16px; overflow: auto; line-height: 1.45;"><code style="background: none; font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;">complexity = self.complexity_estimator(x).mean()
dynamic_top_k = max(1, min(self.top_k, int(self.top_k * complexity * self.capacity_factor)))</code></pre>

  <h3 style="margin-top: 1.8rem; font-weight: 600;">3. 诊断工具适配与优化</h3>
  <p>
    继承 YOLO-Master 的
    <code style="background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px;">ExpertUsageTracker</code>
    诊断工具，并针对 YOLOv26 的模型结构进行了 Hook 机制优化，使其能够稳定生成专家使用热图与负载统计。
  </p>

  <h3 style="margin-top: 1.8rem; font-weight: 600;">4. YOLOv26 完整集成</h3>
  <p>
    将上述 MoE 模块无缝嵌入 YOLOv26 的骨干网络（Backbone）中，提供了可直接训练的配置文件与脚本。
  </p>
</div>

<!-- 实验结果区块 -->
<div style="margin-bottom: 2.5rem;">
  <h2 style="border-bottom: 1px solid #eaecef; padding-bottom: 0.3rem; margin-top: 1.5rem;">📊 初步实验结果（VOC 2007，50 epoch）</h2>
  <table style="border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.95rem;">
    <thead>
      <tr style="background-color: #f6f8fa; border-bottom: 2px solid #d0d7de;">
        <th style="padding: 10px 12px; text-align: left;">模型</th>
        <th style="padding: 10px 12px; text-align: left;">参数量</th>
        <th style="padding: 10px 12px; text-align: left;">FLOPs</th>
        <th style="padding: 10px 12px; text-align: left;">mAP50 (VOC val)</th>
        <th style="padding: 10px 12px; text-align: left;">备注</th>
      </tr>
    </thead>
    <tbody>
      <tr style="border-bottom: 1px solid #eaecef;">
        <td style="padding: 10px 12px;">YOLOv26（原体）</td>
        <td style="padding: 10px 12px;">2.51M</td>
        <td style="padding: 10px 12px;">5.8G</td>
        <td style="padding: 10px 12px;">0.245</td>
        <td style="padding: 10px 12px;">基线，训练 50 轮</td>
      </tr>
      <tr style="border-bottom: 1px solid #eaecef;">
        <td style="padding: 10px 12px;">YOLO_SparseDualMoE（本仓库）</td>
        <td style="padding: 10px 12px;">2.33M</td>
        <td style="padding: 10px 12px;">5.4G</td>
        <td style="padding: 10px 12px;">0.251</td>
        <td style="padding: 10px 12px;">精度相近，资源略降</td>
      </tr>
    </tbody>
  </table>
  <div style="background-color: #f8f9fa; border-left: 4px solid #6c757d; padding: 0.8rem 1.2rem; margin: 1.2rem 0; border-radius: 0 6px 6px 0;">
    <p style="margin: 0;">
      <strong>说明</strong>：以上结果仅基于单次训练，未经过充分调参。由于训练数据有限（VOC 2007 约 4k 训练图），仅做集成可行性参考，
      <strong>结论尚需更多实验验证</strong>。
    </p>
  </div>
</div>

<!-- 快速开始区块 -->
<div style="margin-bottom: 2.5rem;">
  <h2 style="border-bottom: 1px solid #eaecef; padding-bottom: 0.3rem; margin-top: 1.5rem;">🚀 快速开始</h2>
  <h3 style="margin-top: 1.5rem; font-weight: 600;">安装依赖</h3>
  <pre style="background-color: #f6f8fa; border-radius: 6px; padding: 16px; overflow: auto; line-height: 1.45;"><code style="background: none; font-family: SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace;">pip install -r requirements.txt</code></pre>
</div>

<!-- 已知问题与未来计划区块 -->
<div style="margin-bottom: 2.5rem;">
  <h2 style="border-bottom: 1px solid #eaecef; padding-bottom: 0.3rem; margin-top: 1.5rem;">🔧 已知问题与未来计划</h2>
  <ul style="padding-left: 1.8rem; line-height: 1.6;">
    <li>在小分辨率特征图（如 2×2）上，复杂度估计器输出接近 0，导致动态 <code style="background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px;">top_k</code> 降为 1，加剧专家负载不均。</li>
    <li>训练早期可能出现数值不稳定（NaN），导致梯度失效。</li>
    <li>路由器使用全局平均池化，丢失空间信息，限制了专家选择的多样性。</li>
    <li>当前版本在保存完整模型 checkpoint 时可能遇到非叶张量错误（官方 <code style="background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px;">_robust_deepcopy</code> 未引入），建议仅保存 <code style="background-color: #f3f4f6; padding: 0.2rem 0.4rem; border-radius: 4px;">state_dict</code> 以规避。</li>
  </ul>
  <p>
    详细分析见
    <a href="docs/ROOT_CAUSE_ANALYSIS.md">docs/ROOT_CAUSE_ANALYSIS.md</a>。
  </p>
  <p style="margin-top: 1.2rem;">
    <strong>未来计划：</strong>
  </p>
  <ul style="padding-left: 1.8rem; line-height: 1.6;">
    <li>引入更精细的负载均衡策略</li>
    <li>探索专家权重共享以降低参数量</li>
    <li>提供 ONNX/TensorRT 部署示例</li>
  </ul>
</div>

<!-- 致谢区块 -->
<div style="margin-bottom: 2.5rem;">
  <h2 style="border-bottom: 1px solid #eaecef; padding-bottom: 0.3rem; margin-top: 1.5rem;">📚 致谢</h2>
  <p>本项目基于以下优秀工作：</p>
  <ul style="padding-left: 1.8rem; line-height: 1.6;">
    <li>
      <a href="https://github.com/isLinXu/YOLO-Master">YOLO-Master</a>：提供了高效的 MoE 基础实现与路由设计。
    </li>
    <li>
      <a href="https://github.com/ultralytics/ultralytics">Ultralytics YOLO</a>：提供了 YOLO 系列的强大训练框架。
    </li>
  </ul>
</div>
