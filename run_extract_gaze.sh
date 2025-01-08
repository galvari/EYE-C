
INPUT_KEYPOINTS_FOLDER=$1
INPUT_VIDEO_FOLDER=$2
OUTPUT_VIDEO_FOLDER=$3
OUTPUT_JSON_FOLDER=$4

GPU_NUMBER=$5

for i in ${INPUT_KEYPOINTS_FOLDER}/*; do
    echo $i

    # path/to/video_id
    id=${i##*/}  # video_id

    CUDA_VISIBLE_DEVICES=1 python src/extract_gaze.py \
        --use_cuda \
        $i \
        ${INPUT_VIDEO_FOLDER}/${id}.mp4 \
        ${OUTPUT_VIDEO_FOLDER} \
        ${OUTPUT_JSON_FOLDER} \
         --use_cuda \
        --model_weights ./models/gaze360_model.pth.tar \
        --maximize_boxes \
        --enlarge_boxes

    echo "-----------------------------------"
    echo
done

# Example run
# ./run_extract_gaze.sh <input_keypoints_folder> <input_video_folder> <output_video_folder> <output_json_folder>
# ./run_extract_gaze.sh ./INPUT/JSON/Example_copycodec/ ./INPUT/Videos/Example_mp4_copycodec/ ./OUTPUT_BBOX5/Render_copycodec/ ./OUTPUT_BBOX5/JSON_copycodec/
