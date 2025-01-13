
INPUT_KEYPOINTS_FOLDER=$1
INPUT_VIDEO_FOLDER=$2
OUTPUT_VIDEO_FOLDER=$3
OUTPUT_JSON_FOLDER=$4


for i in ${INPUT_KEYPOINTS_FOLDER}/*; do
    echo $i

    # path/to/video_id
    id=${i##*/}  # video_id

    # Check for .mp4 and .wmv files
    if [ -f "${INPUT_VIDEO_FOLDER}/${id}.mp4" ]; then
        video_file="${id}.mp4"
    elif [ -f "${INPUT_VIDEO_FOLDER}/${id}.wmv" ]; then
        video_file="${id}.wmv"
    else
        echo "No video file found for ${id} in ${INPUT_VIDEO_FOLDER}"
        continue  # Skip this iteration if no video file is found
    fi

    echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} python src/extract_gaze.py \
        --use_cuda \
        $i \
        ${INPUT_VIDEO_FOLDER}/${video_file} \
        ${OUTPUT_VIDEO_FOLDER} \
        ${OUTPUT_JSON_FOLDER} \
        --model_weights ./models/gaze360_model.pth.tar \
        --enlarge_boxes

    echo "-----------------------------------"
    echo
done


# Example run
# ./run_extract_gaze.sh <input_keypoints_folder> <input_video_folder> <output_video_folder> <output_json_folder>
