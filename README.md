# 3D Projection

该项目用于处理 `stl` 或 `ply` 三维模型文件，复用 `3D_model_Gm_final` 中的关键预处理思路，完成以下流程：

- 输入 `stl/ply` 文件
- 统一转换为 `ply` 点云格式
- 点云去噪
- 姿态矫正到 `Z` 轴
- 上下端（基部）判定与翻转
- 基部锚定到原点附近
- 将点投影到 `XY` 平面
- 生成基于点密度的灰度投影结果
- 将 `stl` 文件单独转换为 `ply` 点云文件

## 环境

项目使用 `uv` 管理依赖。

```bash
cd .
uv sync --extra dev
```

## 命令行参数

- `--input` / `-i`：必填。输入文件或输入目录。
- `--output-dir` / `-o`：可选。输出目录，默认是项目根目录下的 `output/`
- `--sample-points`：仅对 `stl` 生效，控制网格采样点数。
- `--grid-size`：密度统计网格大小。
- `--image-size`：投影图输出尺寸。
- `--no-log-density`：关闭密度图的对数增强。
- `--nb-neighbors`、`--std-ratio`、`--black-filter`、`--black-threshold`、`--slicing-ratio`：预处理相关参数。
- `--rotate-x-deg`、`--rotate-y-deg`、`--rotate-z-deg`：手动微调旋转角度，单位为度。
- `--interactive-rotate`：单文件交互式旋转调参模式。
- `--adjustment`：兼容旧逻辑的 `Y` 轴 `90°` 步进旋转参数。

## 用法

### 处理整个目录

批量处理目录中的所有 `ply/stl` 文件：

```bash
cd .
uv run pointcloud-project \
  --input /path/to/input_dir
```

如果输入目录与项目同级，也可以写成相对于项目根目录的路径：

```bash
uv run pointcloud-project \
  --input ../3D_model_Gm_final_input
```

### 处理单个文件

```bash
uv run pointcloud-project \
  --input /path/to/model.ply
```

或：

```bash
uv run pointcloud-project \
  --input /path/to/model.stl
```

### 指定输出目录和参数

```bash
uv run pointcloud-project \
  --input /path/to/input_dir \
  --output-dir ./output \
  --grid-size 512 \
  --image-size 1024
```

### 输出不正确时的两种调整方式

如果某个样本姿态没有矫正到理想方向，可以选择以下两种方式：

1. 直接输入旋转角度

```bash
uv run pointcloud-project \
  --input /path/to/model.ply \
  --rotate-x-deg 5 \
  --rotate-y-deg -8 \
  --rotate-z-deg 0
```

2. 进入交互式调参模式

```bash
uv run pointcloud-project \
  --input /path/to/model.ply \
  --interactive-rotate
```

交互模式只支持单文件输入。运行后程序会先生成预览图，然后在终端中持续等待命令。

支持的命令如下：

- `x <deg>`：在当前基础上给 `x` 轴增加角度
- `y <deg>`：在当前基础上给 `y` 轴增加角度
- `z <deg>`：在当前基础上给 `z` 轴增加角度
- `set <x> <y> <z>`：直接设置三轴角度
- `show`：按当前角度重新生成预览
- `reset`：恢复到 `0/0/0`
- `save`：接受当前角度并正式输出结果
- `quit`：退出交互模式且不保存

## STL 转 PLY

项目中额外提供了一个独立目录 `stl_to_ply_output/`，专门保存 `stl -> ply` 的转换结果。

对应命令为：

```bash
uv run stl-to-ply \
  --input /path/to/model.stl
```

如果要批量转换整个目录中的所有 `stl` 文件：

```bash
uv run stl-to-ply \
  --input /path/to/stl_dir
```

如果输入目录与项目同级，也可以写成相对路径：

```bash
uv run stl-to-ply \
  --input ../some_stl_dir \
  --output-dir stl_to_ply_output
```

该命令会将每个 `stl` 网格采样成点云，并输出：

- `*.ply`：转换后的点云文件
- `*_stl_to_ply_summary.json`：当前文件的转换摘要
- `batch_summary.json`：整批转换汇总

## 输出结果

对每个输入文件，程序会生成：

- `*_converted_input.ply`：统一转换后的输入点云
- `*_cleaned.ply`：去噪并居中的点云
- `*_aligned.ply`：姿态矫正后的点云
- `*_projected_xy.ply`：投影到 `XY` 平面的点云
- `*_projection.png`：二维投影图
- `*_density_xy.ply`：带灰度密度信息的投影点云
- `*_density.png`：灰度密度图
- `*_summary.json`：当前文件的处理参数、统计信息和输出路径

如果输入的是一个目录，还会额外生成：

- `batch_summary.json`：整批文件的处理汇总

## 项目目录结构说明

仓库的主要目录和文件如下：

```text
3D_projection/
├── src/
│   └── pointcloud_projection/
│       ├── __init__.py              # 包初始化文件
│       ├── cli.py                   # 主流程 CLI，负责预处理、投影和批处理编排
│       ├── io.py                    # `ply/stl` 读取与点云写出
│       ├── paths.py                 # 项目根目录路径解析与相对路径格式化
│       ├── preprocess.py            # 去噪、姿态矫正、上下端判定与基部锚定
│       ├── projection.py            # XY 投影、密度统计、灰度图与密度点云生成
│       └── stl_to_ply.py            # `stl -> ply` 转换入口，支持单文件和目录批处理
├── tests/
│   ├── test_cli.py                  # 主流程 CLI 与交互命令解析测试
│   ├── test_io.py                   # 输入统一转 PLY 的测试
│   ├── test_paths.py                # 路径与导入时间排序测试
│   ├── test_preprocess.py           # PCA 主轴与手动旋转测试
│   ├── test_projection.py           # 投影与密度计算相关测试
│   └── test_stl_to_ply.py           # `stl -> ply` 转换相关测试
├── output/                          # 主流程默认输出目录
├── stl_to_ply_output/               # `stl -> ply` 转换结果目录
├── pyproject.toml                   # uv 依赖配置与命令行入口定义
└── README.md                        # 项目说明文档
```

各模块职责如下：

- `src/pointcloud_projection/` 负责项目核心实现，适合直接复用或继续扩展
- `cli.py` 负责主流程命令行入口，组织输入遍历、预处理、投影和结果汇总
- `io.py` 负责模型读取、统一转 `ply` 和点云写出，是 `ply/stl` 输入输出的基础模块
- `preprocess.py` 负责复用 `3D_model_Gm_final` 的预处理逻辑，并用 PCA 主轴完成姿态对齐
- `projection.py` 负责把点云投影到 `XY` 平面，并生成灰度密度结果
- `stl_to_ply.py` 负责独立的 `stl -> ply` 转换流程，不参与投影处理
- `paths.py` 负责把路径统一解析到项目根目录
- `tests/` 负责自动化验证投影流程、交互命令、PCA 对齐和 `stl -> ply` 转换流程能否正常运行
- `output/` 负责保存主流程生成的投影结果
- `stl_to_ply_output/` 负责保存 `stl -> ply` 转换结果

## 路径规则

项目中的路径统一按“项目根目录”解释：

- `--input` 如果传相对路径，则相对于项目根目录解析
- `--output-dir` 如果不传，默认使用项目根目录下的 `output/`
- 输出的 `summary.json` 和 `batch_summary.json` 中，路径也会以相对于项目根目录的形式记录
- `stl-to-ply` 命令默认使用项目根目录下的 `stl_to_ply_output/`

例如在项目根目录执行时：

```bash
uv run pointcloud-project \
  --input ../3D_model_Gm_final_input \
  --output-dir output
```

## 常见问题

### 1. 为什么必须传 `--input`？

当前工具不再内置默认输入目录，目的是避免把路径写死在程序里。使用时需要显式指定输入文件或输入目录。

### 2. 输入目录里可以同时放 `ply` 和 `stl` 吗？

可以。程序会自动遍历目录中的 `ply` 和 `stl` 文件，并逐个处理。

### 3. 为什么 `stl` 输入处理会更慢？

因为 `stl` 本身是三角网格，程序会先把网格采样成点云，再执行后续预处理和投影，所以通常比直接读取 `ply` 更耗时。

### 4. `projection.png` 和 `density.png` 有什么区别？

- `projection.png`：纯二维投影结果，只表示点投到 `XY` 平面后的位置
- `density.png`：按 `XY` 平面上的局部点数生成灰度图，越亮表示点越密集

### 5. `density_xy.ply` 里保存的是什么？

它保存的是投影到 `XY` 平面后的密度点云，点的颜色使用灰度表示对应区域的点密度。

### 6. 如果输出图太稀疏或太粗糙怎么办？

可以尝试调整：

- `--grid-size`：增大后，密度统计会更细
- `--image-size`：增大后，输出图片分辨率更高
- `--sample-points`：对 `stl` 输入增大采样点数，可以保留更多细节

### 7. 如果我只想做 `stl -> ply` 转换，不想跑完整投影流程怎么办？

直接使用：

```bash
uv run stl-to-ply --input /path/to/model.stl
```

这个命令只负责把 `stl` 网格采样为 `ply` 点云，不做姿态矫正和投影。

### 8. 交互式旋转模式适合什么场景？

当某些样本经过自动 PCA 对齐后，投影方向仍然不理想时，可以用 `--interactive-rotate` 单独调一个文件，边看预览边微调角度。

## 实现说明

- `ply` 输入会直接按点云读取。
- `stl` 输入会先读取为三角网格，再采样为点云，并统一输出一个 `converted_input.ply` 后进入相同流程。
- 灰度越亮，表示 `XY` 平面对应区域的点越密集。
- 预处理核心逻辑来自 `3D_model_Gm_final/preprocess_3d_model.py`，当前项目对其进行了模块化封装，便于独立运行、批处理和交互式调参。
