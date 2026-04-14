<div class="yolo-moe-project">
  <!-- 项目标题与徽章 -->
  <div class="project-header">
    <h1>YOLO-MoE: Lightweight Mixture-of-Experts for YOLOv26</h1>
    <p><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"> <a href="https://opensource.org/licenses/MIT">License Link</a></p>
  </div>

  <!-- 摘要 -->
  <div class="abstract">
    <blockquote>
      <p>本项目将 Mixture-of-Experts (MoE) 轻量化后移植到 YOLOv26 目标检测模型，探索条件计算在边缘检测模型上的可行性。<strong>目前仍处于实验阶段</strong>，存在负载不均衡、数值不稳定等问题，后续将逐步改进。</p>
    </blockquote>
  </div>

  <!-- 项目动机 -->
  <div class="section motivation">
    <h2>🎯 项目动机</h2>
    <p>YOLOv26 为边缘设备设计，但密集卷积层无论输入内容如何都消耗固定计算资源。MoE 通过稀疏激活专家实现条件计算，理论上可降低平均开销。然而，直接将现有 MoE 模块（如 YOLO_Master）移植到 YOLOv26 会导致参数量大幅增加、训练时间延长。为此，我们设计了轻量化 <code>SparseDualMoE</code> 模块，采用共享专家 + 稀疏激活专家的结构。</p>
  </div>

  <!-- 核心特性 -->
  <div class="section features">
    <h2>✨ 核心特性</h2>
    <ul>
      <li><strong>SparseDualMoE</strong>：共享专家（始终激活）+ 稀疏专家（Top‑K 选择）+ 可学习融合门控</li>
      <li><strong>UltraEfficientRouter</strong>：深度可分离卷积 + 8 倍下采样，<strong>路由部分</strong> FLOPs 降低约 95%</li>
      <li><strong>InvertedResidualExpert</strong>：MobileNetV2 风格，参数量少、计算快</li>
      <li><strong>辅助损失</strong>：负载均衡损失 + Z‑Loss + 熵正则（支持分布式训练）</li>
      <li><strong>诊断工具</strong>：自动 Hook 路由输出，生成专家使用率热图、负载变异系数</li>
      <li><strong>剪枝工具</strong>：基于使用率移除低效专家（实验性）</li>
      <li><strong>超参数搜索</strong>：集成 Optuna，自动调优 MoE 参数（仅小规模测试）</li>
    </ul>
  </div>

  <!-- 实验结果 -->
  <div class="section results">
    <h2>📊 初步实验结果（VOC 2007，50 epoch）</h2>
    <table>
      <thead>
        <tr>
          <th>模型</th>
          <th>参数量</th>
          <th>FLOPs</th>
          <th>mAP50 (VOC val)</th>
          <th>备注</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>YOLOv26（原体）</td>
          <td>2.51M</td>
          <td>5.8G</td>
          <td>0.245</td>
          <td>基线，训练 50 轮</td>
        </tr>
        <tr>
          <td>YOLO_SparseDualMoE（本仓库）</td>
          <td>2.33M</td>
          <td>5.4G</td>
          <td>0.251</td>
          <td>精度相近，资源略降</td>
        </tr>
      </tbody>
    </table>
    <div class="note">
      <p><strong>说明</strong>：以上结果仅基于单次训练，未经过充分调参。由于训练数据有限（VOC 2007 约 4k 训练图），且模型仍存在负载不均等问题，<strong>结论尚需更多实验验证</strong>。</p>
    </div>
  </div>

  <!-- 已知问题 -->
  <div class="section known-issues">
    <h2>🚧 已知问题与局限性</h2>
    <ul>
      <li>在小分辨率特征图（如 2×2）上，复杂度估计器输出接近 0，导致动态 <code>top_k</code> 降为 1，加剧专家负载不均。</li>
      <li>训练早期可能出现数值不稳定（NaN），导致梯度失效。</li>
      <li>路由器使用全局平均池化，丢失空间信息，限制了专家选择的多样性。</li>
      <li>异构专家（kernel 3/5）在小特征图上表现不一致，可能引入选择偏差。</li>
      <li>平衡辅助损失在 warmup 后系数过低，对负载均衡的约束减弱。</li>
    </ul>
    <p>详细分析见 <a href="docs/ROOT_CAUSE_ANALYSIS.md">docs/ROOT_CAUSE_ANALYSIS.md</a>。</p>
  </div>

  <!-- 快速开始 -->
  <div class="section quick-start">
    <h2>🚀 快速开始</h2>
    
    <h3>安装依赖</h3>
    <pre><code class="language-bash">pip install -r requirements.txt</code></pre>
  </div>
</div>
