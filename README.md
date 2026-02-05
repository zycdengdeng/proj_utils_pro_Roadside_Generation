### 这里是专门给老年痴呆写的readme，老奶奶看了都用的明白

## 主要流程： 启动环境conda activate zihanw -> 控制帧图像生成方法 -> 控制头视频生成方法 -> caption替换方法

## 控制帧图像生成方法

/mnt/zihanw/proj_utils_pro_Roadside/run_interactive.sh

可用的投影项目:
  1) basic        - 基本路侧merge激光雷达点云投影
  2) blur         - blur投影（路侧相机着色merge激光雷达点云投影）
  3) blur_dense   - blur稠密化投影（路端merge点云投影完规则稠密化）
  4) depth        - depth投影（merge点云生成深度）
  5) depth_dense  - depth稠密化投影（merge点云生成深度后规则稠密化）
  6) hdmap        - HDMap投影（3D→2D bbox，实心填充）

  7) batch        - 批量处理（选择多个项目串行执行）

### 交互流程（已更新）

交互现在为**逐个场景输入**，每个场景可以指定**多个车辆ID**：

```
步骤 1/2: 逐个输入场景和目标车辆ID

--- 第 1 个场景 ---
请输入场景ID（直接Enter结束）: 004
请输入车辆ID [默认45]: 45 67
✓ 场景 004 → 车辆ID: [45, 67]

--- 第 2 个场景 ---
请输入场景ID（直接Enter结束）: 005
请输入车辆ID [默认45]: 19 3
✓ 场景 005 → 车辆ID: [19, 3]

--- 第 3 个场景 ---
请输入场景ID（直接Enter结束）:    ← 直接Enter结束输入

步骤 2/2: 选择批次模式
```

- **场景ID**：直接填 001-089 的编号，会自动找到对应的数据路径
- **车辆ID**：从路侧动态标注文件中读取该ID车辆的位姿，计算 world2lidar 变换。可以空格分隔输入多个ID
- **不再需要 world2lidar_transforms.json**，位姿直接从标注文件实时计算
- 输入完场景后直接按 Enter（不输入场景ID）就结束输入
- 线程数等配置直接 enter 使用默认值

### 输出目录结构

输出目录格式为 `{场景ID}_id{车辆ID}`，例如：

```
blur投影/
  004_id45/        ← 场景004投影到车辆45
  004_id67/        ← 场景004投影到车辆67
  005_id19/        ← 场景005投影到车辆19
```

每个目录下按时间戳分帧存储，每帧可以验证质量。

### 运行方式

1. 在任意目录下都可以执行主脚本（绝对路径），脚本使用 `Path(__file__).resolve().parent` 自动解析路径
2. 运行 `run_interactive.sh`，1-6 是单个功能，7 是批量处理多个功能
3. 选了 7 之后多选功能，如需 blur + depth + HDMap 就选择 2 4 6
4. 逐个输入场景ID和车辆ID，配置直接 enter 使用默认，跑就行了
5. HDMap 的自车排除ID = 投影目标车辆ID（自动处理，不需要额外输入）

## 控制头视频生成方法
这步的时候一开始要选7
1. 脚本在这里 /mnt/zihanw/proj_utils_pro_Roadside/transfer_video_maker/generate_videos.sh
2. 一开始的菜单同上，一般来讲直接选Batch就可以，然后选择自己想要的功能。
3. 关于每个seg的帧数，seg的数量是3 和视频帧率29， 这些默认值是符合cosmos post training demo的配置，默认1280 720p，生成在transfer_video_maker/output，主要分类依据不是场景号，是根据transfer控制头分类的。

## caption替换方法
1. 执行脚本 transfer_video_maker/generate_transfer2_videos.py
2. 如果是替换单个头，就选单个，然后选择适应这个头的选项，每个选项都说明自己是哪个控制头的caption
3. 如果替换多个头，直接选替换全部，然后会多一个自适应匹配的选项，选那个就可以
4. 想替换预设caption直接改py脚本里面的txt就可以了

选择话，看这个例子
找到 3 个数据集:

BlurProjection (147 个caption文件)
DepthSparse (147 个caption文件)
HDMapBbox (147 个caption文件)
全部数据集
退出
请选择数据集 [1-4, 0]: 4

已选择: 全部 3 个数据集

总计: 441 个caption文件

预设caption模板:
0) 自动匹配（每个数据集使用对应的预设模板）

unified: "{view_prefix}. The ego vehicle is traveling from {direction}..."
depth: "{view_prefix}. The ego vehicle is traveling from {direction}..."
自定义模板
请选择模板 [0-3]: 0

使用自动匹配模式：每个数据集使用对应的预设模板

匹配结果:
BlurProjection → unified
DepthSparse → unified
HDMapBbox → unified
这个是对的

### 仅限于学习交流，商业用途可以给我磕头笑鼠
