#!//bin/bash
# Enable strict error handling:
# -e: Exit immediately if a command exits with a non-zero status.
# -u: Treat unset variables as an error and exit immediately.
# -o pipefail: Return the exit status of the last command in the pipeline that failed.
set -euo pipefail

if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "Virtual‑env not found (.venv/bin/activate). Aborting." >&2
    exit 1
fi

# ---------------- Default values -----------------
DATA_MODE=""
NUM_EPOCHS=0
CLI_NUM_EPOCHS=0
USE_PATCHES=true
REPATCH=false
NUM_PATCHES=0
PATCH_SIZE=0
CLI_PATCH_SIZE=0
LOG_FILE="run_log.txt"
INTERACTIVE=false
APP_MODE="afgsa"
DETERMINISTIC=false
CURVE_ORDER="raster"
USE_LPIPS_LOSS=false
LIPPS_LOSS_W=0.0
USE_SSIM_LOSS=false
SSIM_LOSS_W=0.0
USE_MULTISCALE_DISCRIMINATOR=false
USE_FILM=false

cleanup() {
    echo "Process interrupted. Cleaning up..."
    exit 130
}
trap cleanup INT

while [[ $# -gt 0 ]]; do
    case "$1" in
    -s | --small)
        DATA_MODE="small"
        shift
        ;;
    -f | --full)
        DATA_MODE="full"
        shift
        ;;
    -l | --limited)
        DATA_MODE="limited"
        shift
        ;;
    -i | --interactive)
        INTERACTIVE=true
        shift
        ;;
    -a | --afgsa)
        APP_MODE="afgsa"
        shift
        ;;
    -m | --mamba)
        APP_MODE="mamba"
        shift
        ;;
    -e | --epochs)
        CLI_NUM_EPOCHS="$2"
        shift 2
        ;;
    -P | --no-patches)
        USE_PATCHES=false
        shift
        ;;
    --patch-size)
        CLI_PATCH_SIZE="$2"
        shift 2
        ;;
    --repatch)
        REPATCH=true
        shift
        ;;
    --deterministic)
        DETERMINISTIC=true
        shift
        ;;
    --raster-curve)
        CURVE_ORDER="raster"
        shift
        ;;
    --hilbert-curve)
        CURVE_ORDER="hilbert"
        shift
        ;;
    --zorder-curve)
        CURVE_ORDER="zorder"
        shift
        ;;
    --lpips-loss=*)
        USE_LPIPS_LOSS=true
        LIPPS_LOSS_W="${1#*=}"
        shift
        ;;
    --ssim-loss=*)
        USE_SSIM_LOSS=true
        SSIM_LOSS_W="${1#*=}"
        shift
        ;;
    --multiscale-discriminator)
        USE_MULTISCALE_DISCRIMINATOR=true
        shift
        ;;
    --use-film)
        USE_FILM=true
        shift
        ;;
    *)
        echo "Unknown option: $1" >&2
        exit 1
        ;;
    esac
done

if [[ -z "$DATA_MODE" ]]; then
    echo "Please specify the dataset mode: -s|--small, -f|--full, -l|--limited" >&2
    exit 1
fi

BASE_DIR="${HOME}/projects/pixel_heal_thyself"
INPUT_DIR="${BASE_DIR}/data/images_${DATA_MODE}"
RUN_ROOT="${BASE_DIR}/runs/${APP_MODE}_${DATA_MODE}"
mkdir -p "${RUN_ROOT}"

get_next_run_directory() {
    local run_root="$1"
    local next_id=1

    if [[ -d "${run_root}" ]]; then
        # Check if any run directories exist
        if compgen -G "${run_root}/run_*" >/dev/null; then
            last=$(ls -d ${run_root}/run_* 2>/dev/null | sort -V | tail -n1)
            if [[ -n "${last}" && "${last##*_}" =~ ^[0-9]+$ ]]; then
                next_id=$((10#${last##*_} + 1))
            else
                next_id=1
            fi
        fi
    else
        echo "Run root directory does not exist. Exiting."
        exit 1
    fi

    printf "%s/run_%03d" "${run_root}" "${next_id}"
}

OUTPUT_DIR="$(get_next_run_directory "${RUN_ROOT}")"
mkdir -p "${OUTPUT_DIR}"
: >"${OUTPUT_DIR}/${LOG_FILE}" # truncate/creates log file

log() { printf '%s\n' "$*" >>"${OUTPUT_DIR}/${LOG_FILE}"; }

case "$DATA_MODE" in
full)
    NUM_EPOCHS=12
    NUM_PATCHES=400
    PATCH_SIZE=128
    ;;
limited)
    NUM_EPOCHS=12
    NUM_PATCHES=200
    PATCH_SIZE=64
    ;;
small)
    NUM_EPOCHS=12
    NUM_PATCHES=100
    PATCH_SIZE=32
    ;;
esac

# CLI overrides
[[ $CLI_NUM_EPOCHS -gt 0 ]] && NUM_EPOCHS=$CLI_NUM_EPOCHS
[[ $CLI_PATCH_SIZE -gt 0 ]] && PATCH_SIZE=$CLI_PATCH_SIZE

# Patches are shared between all modes
# We will generate a name for the dir to save them based on the size and count
PATCHES_ROOT="${INPUT_DIR}/patches"
PATCHES_DIR="${PATCHES_ROOT}/${PATCH_SIZE}x${PATCH_SIZE}-n${NUM_PATCHES}"
if [[ ${REPATCH} == true ]]; then
    rm -rf "${PATCHES_DIR}"
fi
mkdir -p "${PATCHES_DIR}"

SEPARATOR=$(printf '─%.0s' {1..100})
log "Training Data:"
log "$SEPARATOR"
log "Base dir    : ${BASE_DIR}"
log "Dataset     : ${DATA_MODE}"
log "Input dir   : ${INPUT_DIR}"
log "Patches dir : ${PATCHES_DIR}"
log "Output dir  : ${OUTPUT_DIR}"
num_images=$(find "${INPUT_DIR}" -type f -name '*.exr' | wc -l)
log "Num images  : ${num_images}"
log "$SEPARATOR"

log
log "Training Parameters:"
log "$SEPARATOR"
log "Epochs        : ${NUM_EPOCHS}"
log "Use patches   : ${USE_PATCHES}"
log "Num patches   : ${NUM_PATCHES}"
log "Patch size    : ${PATCH_SIZE}"
log "Interactive   : ${INTERACTIVE}"
log "App mode      : ${APP_MODE}"
log "Deterministic : ${DETERMINISTIC}"
log "Curve order   : ${CURVE_ORDER}"
log "Repatch       : ${REPATCH}"
log "LPIPS loss    : ${USE_LPIPS_LOSS}"
log "LPIPS loss W  : ${LIPPS_LOSS_W}"
log "SSIM loss     : ${USE_SSIM_LOSS}"
log "SSIM loss W   : ${SSIM_LOSS_W}"
log "Multiscale D  : ${USE_MULTISCALE_DISCRIMINATOR}"
log "Use FILM      : ${USE_FILM}"
log "$SEPARATOR"

CMD="uv run pht/models/afgsa/code/wo_diff_spec_decomp/train/train.py \
  --inDir ${INPUT_DIR} \
  --datasetDir ${PATCHES_DIR} \
  --outDir ${OUTPUT_DIR} \
  --epochs ${NUM_EPOCHS} \
  --numPatches ${NUM_PATCHES} \
  --patchSize ${PATCH_SIZE} \
  --curveOrder ${CURVE_ORDER}"

if false; then
    CMD+=" --appMode ${APP_MODE}"
fi

if [[ ${DETERMINISTIC} == true ]]; then
    CMD+=" --deterministic"
fi

if [[ ${USE_LPIPS_LOSS} == true ]]; then
    CMD+=" --useLPIPSLoss"
    CMD+=" --lpipsLossW ${LIPPS_LOSS_W}"
fi
if [[ ${USE_SSIM_LOSS} == true ]]; then
    CMD+=" --useSSIMLoss"
    CMD+=" --ssimLossW ${SSIM_LOSS_W}"
fi

if [[ ${USE_MULTISCALE_DISCRIMINATOR} == true ]]; then
    CMD+=" --useMultiscaleDiscriminator"
    # CMD+=" --lrD 1e-4 --lrG 2e-5 --gpLossW 5 --batchSize 16"
fi

if [[ ${USE_FILM} == true ]]; then
    CMD+=" --useFilm"
fi

if [[ ${USE_PATCHES} == false ]]; then
    CMD+=" --useFullImage"
fi
if [[ ${APP_MODE} == "mamba" ]]; then
    CMD+=" --lrD 1e-5 --lrG 1e-4 --ganLossW 1e-4 --gpLossW 1"
fi

log
log "Training:"
log "$SEPARATOR"
log "Command: ${CMD}"

START_TIME=$(date +%s)
log "Start time: $(date -d "@${START_TIME}" '+%Y-%m-%d %H:%M:%S')"

if [[ ${INTERACTIVE} == false ]]; then
    export PYTHONUNBUFFERED=1
    eval "${CMD}" 2>&1 | uv run tools/termlog.py "${OUTPUT_DIR}/train.txt"
    export PYTHONUNBUFFERED=0
else
    eval "${CMD}"
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
log "End time  : $(date -d "@${END_TIME}" '+%Y-%m-%d %H:%M:%S')"
log "Elapsed   : ${ELAPSED} seconds"
log "$SEPARATOR"
