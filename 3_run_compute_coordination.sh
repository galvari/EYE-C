

INPUT_JSON_FOLDER=$1
OUTPUT_JSON_FOLDER=$2
COORDINATION_FACTOR=$3
Z_FACTOR=$4
INPUT_VIDEO=$5
OUTPUT_VIDEO_FOLDER=$6


for i in ${INPUT_JSON_FOLDER}/*; do
    echo $i
    id=${i##*/}
    v_id=${id%_*}


    python src/estimate_coordination.py \
        $i \
        $OUTPUT_JSON_FOLDER \
        --coordination_factor $COORDINATION_FACTOR \
        --z_factor $Z_FACTOR \
        --input_video ${INPUT_VIDEO}/${v_id}_gaze.wmv \
        --output_video_folder ${OUTPUT_VIDEO_FOLDER}
 
    echo "-----------------------------------"
    echo
done

# Example run
# ./run_compute_coordination_3.sh .../JSON .../dist08_z03 0.8 0.3 .../VideoRendered .../dist08_z03
