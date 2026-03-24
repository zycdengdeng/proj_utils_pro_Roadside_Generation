#!/bin/bash
# Transfer2 视频生成（segment 模式）
# 交互选择投影类型和场景，然后批量生成

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SEGMENTS_DIR="${PROJECT_ROOT}/segment_pipeline/output"
OUTPUT_BASE="${SCRIPT_DIR}/output"
FPS=10

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Transfer2 视频数据集生成器${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ---- 检查 segments 目录 ----
if [ ! -d "$SEGMENTS_DIR" ]; then
    echo -e "${RED}错误: segment_pipeline 输出目录不存在: ${SEGMENTS_DIR}${NC}"
    echo "请先运行: python -m segment_pipeline.segment_pipeline --interactive"
    exit 1
fi

# ---- 1. 选择投影类型 ----
echo -e "${GREEN}选择投影类型（多选，空格或逗号分隔）:${NC}"
echo "  1) blur           - Blur投影（路侧着色）"
echo "  2) depth          - 深度图投影"
echo "  3) hdmap          - HDMap bbox投影"
echo "  4) depth_dense    - 深度图稠密化投影"
echo "  5) blur_dense     - Blur稠密化投影"
echo "  6) basic          - 基本点云投影"
echo ""
echo -e "  ${YELLOW}默认: 1 2 3 (blur + depth + hdmap)${NC}"
echo ""
read -p "请选择 [直接回车用默认]: " type_choices

# 默认 blur + depth + hdmap
if [ -z "$type_choices" ]; then
    type_choices="1 2 3"
fi
type_choices=${type_choices//,/ }

# 映射选择到投影类型
declare -a PROJECT_TYPES=()
declare -A TYPE_OUTPUT_NAME=(
    [blur]="BlurProjection"
    [depth]="DepthSparse"
    [hdmap]="HDMapBbox"
    [depth_dense]="DepthDense"
    [blur_dense]="BlurDense"
    [basic]="BasicProjection"
)

for c in $type_choices; do
    case $c in
        1) PROJECT_TYPES+=("blur") ;;
        2) PROJECT_TYPES+=("depth") ;;
        3) PROJECT_TYPES+=("hdmap") ;;
        4) PROJECT_TYPES+=("depth_dense") ;;
        5) PROJECT_TYPES+=("blur_dense") ;;
        6) PROJECT_TYPES+=("basic") ;;
        *) echo -e "${RED}忽略无效选择: $c${NC}" ;;
    esac
done

if [ ${#PROJECT_TYPES[@]} -eq 0 ]; then
    echo -e "${RED}错误: 没有选择任何投影类型${NC}"
    exit 1
fi

echo ""
echo -e "已选择: ${GREEN}${PROJECT_TYPES[*]}${NC}"
echo ""

# ---- 2. 选择场景 ----
# 扫描 segments 目录，提取场景号（seg名格式: {scene}_id{vid}_seg{NN}）
ALL_SCENES=$(ls -d "$SEGMENTS_DIR"/*/ 2>/dev/null | xargs -I{} basename {} | sed 's/_id.*$//' | sort -u)

if [ -z "$ALL_SCENES" ]; then
    echo -e "${RED}错误: ${SEGMENTS_DIR} 下没有找到任何 segment 目录${NC}"
    exit 1
fi

echo -e "${GREEN}可用场景:${NC}"
echo -e "  ${ALL_SCENES}" | tr '\n' '  '
echo ""
echo ""
echo -e "  ${YELLOW}默认: all (全部场景)${NC}"
echo ""
read -p "输入场景号（空格分隔，或直接回车选all）: " scene_input

# 默认 all
if [ -z "$scene_input" ] || [ "$scene_input" = "all" ]; then
    SELECTED_SCENES="$ALL_SCENES"
    echo -e "已选择: ${GREEN}全部场景${NC}"
else
    SELECTED_SCENES="$scene_input"
    echo -e "已选择: ${GREEN}${SELECTED_SCENES}${NC}"
fi

echo ""

# ---- 3. 根据选择的场景过滤 seg 目录 ----
filter_segs_by_scenes() {
    local segments_dir="$1"
    shift
    local scenes=("$@")

    for seg_dir in "$segments_dir"/*/; do
        [ -d "$seg_dir" ] || continue
        seg_name=$(basename "$seg_dir")
        scene_id="${seg_name%%_id*}"
        for s in "${scenes[@]}"; do
            if [ "$scene_id" = "$s" ]; then
                echo "$seg_name"
                break
            fi
        done
    done
}

SCENE_ARRAY=($SELECTED_SCENES)
MATCHING_SEGS=$(filter_segs_by_scenes "$SEGMENTS_DIR" "${SCENE_ARRAY[@]}")
SEG_COUNT=$(echo "$MATCHING_SEGS" | grep -c . || true)

echo -e "匹配的 segment 数量: ${GREEN}${SEG_COUNT}${NC}"
echo -e "投影类型数量: ${GREEN}${#PROJECT_TYPES[@]}${NC}"
echo -e "总视频组数: ${GREEN}$((SEG_COUNT * ${#PROJECT_TYPES[@]}))${NC} (每组 7 相机)"
echo ""

# ---- 4. 确认并执行 ----
read -p "开始生成? [Y/n]: " confirm
if [[ "$confirm" =~ ^[nN] ]]; then
    echo "已取消"
    exit 0
fi

echo ""

# 为每种投影类型执行生成
FAIL_COUNT=0
for ptype in "${PROJECT_TYPES[@]}"; do
    output_name="${TYPE_OUTPUT_NAME[$ptype]}"
    output_dir="${OUTPUT_BASE}/${output_name}"

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}  生成: ${ptype} → ${output_name}${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # 如果选了特定场景，用临时目录软链接过滤
    if [ "$scene_input" != "" ] && [ "$scene_input" != "all" ]; then
        # 创建临时目录，只链接选中场景的 seg
        TMP_DIR=$(mktemp -d)
        trap "rm -rf $TMP_DIR" EXIT
        for seg_name in $MATCHING_SEGS; do
            ln -s "$SEGMENTS_DIR/$seg_name" "$TMP_DIR/$seg_name"
        done
        seg_source="$TMP_DIR"
    else
        seg_source="$SEGMENTS_DIR"
    fi

    python3 "${SCRIPT_DIR}/generate_transfer2_videos.py" \
        --segments-dir "$seg_source" \
        --project-type "$ptype" \
        --output-dir "$output_dir" \
        --fps "$FPS" \
    && echo -e "${GREEN}  ✓ ${ptype} 完成${NC}" \
    || { echo -e "${RED}  ✗ ${ptype} 失败${NC}"; FAIL_COUNT=$((FAIL_COUNT + 1)); }

    # 清理临时目录
    if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
        rm -rf "$TMP_DIR"
        trap - EXIT
        TMP_DIR=""
    fi

    echo ""
done

echo -e "${BLUE}========================================${NC}"
if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}  全部完成！${NC}"
else
    echo -e "${YELLOW}  完成（${FAIL_COUNT} 个失败）${NC}"
fi
echo -e "${BLUE}========================================${NC}"
echo ""
echo "输出目录: ${OUTPUT_BASE}/"
for ptype in "${PROJECT_TYPES[@]}"; do
    echo "  ${TYPE_OUTPUT_NAME[$ptype]}/"
done
