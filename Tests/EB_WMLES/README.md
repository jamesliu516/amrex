# EB_WMLES — Embedded-Boundary Wall-Modeled Large Eddy Simulation (WMLES) solver

基于 AMReX 的嵌入式边界（EB）可压缩 WMLES 求解器骨架。代码以
`Tests/EB_CNS` 为模板复制并全面重命名（`CNS_*` → `WMLES_*`），目标是使用
WMLES 方法（SGS 亚格子模型 + 壁面模型）对含复杂几何（STL 导入）的湍流开展
模拟，几何采用 AMReX 的 embedded boundary（EB）框架表示。

## 1. 目录结构

```
Tests/EB_WMLES/
├── CMakeLists.txt              # 2D/3D 测试注册（setup_test）
├── Source/                     # 求解器核心（与 Exec 分离，便于多算例复用）
│   ├── main.cpp                # 驱动：Amr 初始化 / 推进 / 输出
│   ├── WMLES.H / WMLES.cpp     # AmrLevel 派生类：状态管理、dt、IO、avgDown 等
│   ├── WMLESBld.cpp            # LevelBld 工厂
│   ├── WMLES_setup.cpp         # 状态量描述符、物理边界条件、导出量
│   ├── WMLES_advance.cpp       # RK2 时间推进（两阶段）
│   ├── WMLES_advance_box.cpp   # 规则网格通量（PLM 重构 + Riemann + 黏性）
│   ├── WMLES_advance_box_eb.cpp# EB 切割单元通量 + 通量重分布
│   ├── WMLES_parm.H / .cpp     # Parm 结构（物性 + SGS + 壁面模型参数）
│   ├── WMLES_sgs_K.H           # Smagorinsky 涡粘 GPU 核（已接入推进循环）
│   ├── WMLES_wallmodel_K.H     # 壁面模型核（占位，Phase 4 实现）
│   ├── WMLES_index_macros.H    # 状态/原始变量索引宏
│   ├── WMLES_K.H               # 初始 dt 估计、温度计算
│   ├── WMLES_bcfill.cpp        # 通用物理边界填充（GNUmake 路径用）
│   ├── WMLES_init_eb2.cpp      # EB2 几何初始化（combustor 隐式曲面；stl 走 EB2::Build）
│   ├── WMLES_derive.*          # 导出量：pressure / x_velocity / ...
│   ├── WMLES_tagging.H         # 密度梯度加密判据
│   ├── hydro/                  # WMLES_hydro_K.H（斜率/Riemann）、hydro_eb_K.H、
│   │                           # WMLES_divop_K.H（EB 散度 + 壁面通量）
│   └── diffusion/              # WMLES_diffusion_K.H / _eb_K.H（Sutherland、常数黏性、SGS）
└── Exec/
    ├── Cylinder/               # 2D（也可 3D）圆柱绕流：隐式曲面圆柱 + 入流边界
    │   ├── inputs / inputs-ci
    │   └── wmles_prob.* / WMLES_bcfill.cpp
    └── STL_Box/                # 3D 仅：STL 导入长方体（演示 STL→EB 全流程）
        ├── make_stl.py         # 生成 ASCII STL 盒（可编辑尺寸）
        ├── box.stl             # 预生成 1.0×0.5×0.5 盒（12 三角面）
        └── inputs / inputs-ci
```

## 2. 数值方法（当前状态）

- **时间推进**：RK2（Hancock 型两阶段），`WMLES_advance.cpp`。
- **对流项**：守恒变量 PLM 重构（`plm_iorder` 1/2、`plm_theta` minmod/MC 限制器）
  + 近似 Riemann 解（Colella-Glaz 型，`hydro/WMLES_hydro_K.H`）。
- **EB 处理**：切割单元散度按面积分数加权并加入壁面通量
  （`hydro/WMLES_divop_K.H`：无滑移假设下用壁面法向二次插值计算黏性壁面通量，
  对流壁面通量为反射型 Riemann 解），随后做**通量重分布**
  （`amrex_flux_redistribute`，`do_reredistribution`，权重 `eb_weights_type`）。
- **分子黏性**：Sutherland 公式（`C_S`、`T_S`）或常数黏性
  （`wmles.mu` = `const_visc_mu` 别名；`const_visc_ki`、`const_lambda` 缺省为
  0 与 `mu*cp/Pr`）。
- **SGS 亚格子模型（已接入，默认开启）**：Smagorinsky
  `ν_t = (Cs·Δ)²·|S|`，`Δ = sgs_filter_width_ratio·(dx·dy·dz)^(1/3)`。
  在扩散系数中叠加 `μ_t = ρ·ν_t`（动量）与 `μ_t·cp/Pr_t`（能量）。
  EB 路径只在**完全规则单元**上叠加 SGS（切割单元留给壁面模型，且避免中心差分
  读到覆盖单元）。开关：`wmles.sgs_enable`。
- **壁面模型（占位）**：`WMLES_wallmodel_K.H` 中的 `wmles_wall_shear_stress`
  目前返回零应力，未接入推进循环——这是 WMLES 的下一步核心工作（见 §5）。
- **AMR**：`refine_cutcells`（切割单元加密）、`refine_boxes`（矩形加密区）、
  密度梯度加密（`refine_dengrad`）；通量寄存器做粗细层守恒。

## 3. 输入参数（`wmles.*` / `prob.*` / `eb2.*`）

| 参数 | 默认 | 说明 |
|---|---|---|
| `wmles.cfl` | 0.3 | CFL 数 |
| `wmles.lo_bc / hi_bc` | 必填 | 0=内, 1=入流, 2=出流, 3=对称, 4=滑移壁, 5=无滑移壁 |
| `wmles.do_visc` | 1 | 是否开启黏性扩散 |
| `wmles.use_const_visc` | 0 | 常数黏性（否则 Sutherland） |
| `wmles.mu` | — | 常数动力黏性（`const_visc_mu` 的别名，给出即自动开启常数黏性） |
| `wmles.const_visc_ki / const_lambda` | 0 / `mu·cp/Pr` | 体黏性 / 热导率 |
| `wmles.Pr` | 0.72 | 分子 Prandtl 数 |
| `wmles.eos_gamma` / `eos_mu` | 1.4 / 28.97 | 比热比 / 平均分子量 |
| `wmles.sgs_enable` | 1 | Smagorinsky SGS 开关 |
| `wmles.Cs_smag` | 0.18 | Smagorinsky 常数 |
| `wmles.Pr_t` | 0.90 | 湍流 Prandtl 数 |
| `wmles.sgs_filter_width_ratio` | 1.0 | 滤波宽度相对网格的系数 |
| `wmles.eb_weights_type` | 0 | 重分布权重（0=1, 1=总能量, 2=质量, 3=体积分数） |
| `wmles.do_reredistribution` | 1 | EB 通量重分布开关 |
| `prob.inflow_velocity / rho / T / p` | — | 均匀入流条件（初始化 + 入流 BC） |
| `eb2.geom_type` | — | `cylinder`（隐式曲面）或 `stl`（STL 文件） |
| `eb2.stl_file / stl_scale / stl_center / stl_reverse_normal / stl_use_bvh` | — | STL 几何参数 |

## 4. 本次已修复的问题（相对骨架提交）

1. **头文件包含名错误**（会导致编译失败）：
   `WMLES.Hydro_K.H` → `WMLES_hydro_K.H`、`WMLES.Hydro_eb_K.H` →
   `WMLES_hydro_eb_K.H`（`WMLES_advance_box.cpp`、`WMLES_advance_box_eb.cpp`、
   `hydro/WMLES_hydro_eb_K.H`）。
2. **`WMLES_K.H` 自我包含**（`#include "WMLES_K.H"`）已删除。
3. **`wmles.mu` 参数此前从未被读取**：现作为 `const_visc_mu` 别名自动开启
   常数黏性；`const_visc_ki`/`const_lambda` 未给出时自动取 0 / `mu·cp/Pr`。
4. **CMake 重复符号风险**：`Source/WMLES_bcfill.cpp` 与各 `Exec/*/WMLES_bcfill.cpp`
   都定义 `wmles_bcfill`，CMake 源列表只保留 Exec 版本（GNUmake 路径仍用 Source
   版本，与 EB_CNS 一致）。
5. **SGS 涡粘模型接入推进循环**（`WMLES_sgs_K.H` 重构出无 flag 的
   `wmles_compute_nu_t_cell` 与 EB 版 `wmles_compute_nu_t`；扩散系数核函数
   `wmles_diffcoef(_eb)` / `wmles_constcoef(_eb)` 增加 `dxinv` 参数并叠加
   `μ_t`、`μ_t·cp/Pr_t`）。
6. **越界防护**：规则网格路径 `qtmp` 增长量由 2 增加到 3，保证 SGS 中心差分
   模板不越界。
7. 输入文件补充 SGS 参数并去掉重复的 `wmles.mu`/`wmles.Pr`。

## 5. WMLES 开发路线图

- [x] **Phase 0**：骨架（EB_CNS 复制、重命名、Cylinder/STL_Box 算例）
- [x] **Phase 1**：编译问题修复 + 参数一致性
- [x] **Phase 2**：Smagorinsky SGS 接入（默认开启，可配置）
- [ ] **Phase 3**：SGS 验证——均匀各向同性湍流（HIT）衰减、槽道湍流（CH）粗网格
  （对比 Smagorinsky / 动态 Smagorinsky / Vreman 等模型）
- [ ] **Phase 4**：壁面模型——平衡对数律（代数）→ ODE 型壁面模型
  （沿壁面法向采样 N 点，求解 `d/dn[(ν+ν_t)du/dn]=0`，
  托马斯三对角求解；在 `compute_diff_wallflux` 中用 `τ_w` 替换无滑移壁面导数），
  并**在切割单元与近壁规则单元上启用**；壁面模型更新间隔
  `wall_model_update_interval`
- [ ] **Phase 5**：圆柱绕流（Re=3.9e3 / 1.4e5）验证分离与再附着、阻力/升力系数
- [ ] **Phase 6**：复杂 STL 几何（真实 CAD 模型）高雷诺数湍流模拟
- [ ] **Phase 7**：GPU 移植（核函数已按 GPU 风格编写，需在 CUDA 后端验证）

## 6. 构建与运行

CMake（推荐，2D/3D 自动注册）：

```bash
cmake -S . -B build -DAMReX_EB=ON -DAMReX_ENABLE_TESTS=ON -DAMReX_SPACEDIM=3
cmake --build build -j N --target Test_WMLES_Cylinder_3d Test_WMLES_STL_Box_3d
# 运行（在对应 RUNTIME_SUBDIR 目录）
./build/Tests/EB_WMLES/Cylinder_3d/Test_WMLES_Cylinder_3d inputs
./build/Tests/EB_WMLES/STL_Box_3d/Test_WMLES_STL_Box_3d inputs
```

更换几何：用 CAD 软件导出 ASCII/Binary STL 替换 `Exec/STL_Box/box.stl`
（或运行 `python make_stl.py` 重新生成），按需调整
`eb2.stl_scale` / `eb2.stl_center` / `eb2.stl_reverse_normal`。

## 7. 已知限制与注意事项

- 壁面模型尚未实现：当前 EB 壁面为“无滑移 + 分子黏性壁面通量”，
  高雷诺数下近壁网格必须足够密（`y+ ~ O(1)`）才会正确，这正是 Phase 4 要解决的。
- SGS 在 2D 下 Smagorinsky 模型退化（缺少涡拉伸），定性可用、定量验证应在 3D。
- `wmles.mu` 为常数黏性；若需 Sutherland，则不设置 `wmles.mu` 并保留
  `wmles.C_S`/`wmles.T_S` 默认值。
- 求解器为显式可压缩格式，马赫数很低时（如不可压圆柱绕流）时间步由声速限制，
  适合入门验证；高雷诺数工程模拟建议后续引入低马赫预处理或半隐式推进。
