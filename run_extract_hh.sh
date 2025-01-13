
INPUT_KEYPOINTS_FOLDER=$1
INPUT_VIDEO_FOLDER=$2
OUTPUT_JSON_FOLDER=$3




for i in ${INPUT_KEYPOINTS_FOLDER}/*; do
    echo $i

    # path/to/video_id
    id=${i##*/}  # video_id

    python src/extract_bbox_size.py \
        $i \
        ${INPUT_VIDEO_FOLDER}/${id}.mp4 \
        ${OUTPUT_JSON_FOLDER}


    echo "-----------------------------------"
    echo
done

