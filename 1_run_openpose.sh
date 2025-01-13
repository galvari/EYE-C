INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

OPENPOSE_BIN=/home/galvari/openpose/build/examples/openpose/openpose.bin
MODEL_FOLDER=/home/galvari/openpose/models/

for i in ${INPUT_FOLDER}/*; do
    echo $i

    # path/to/video_name.mp4
    out=${i%.*}  # path/to/video_name
    out=${out##*/}  # video_name


    echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1} $OPENPOSE_BIN \
        --video $i \
        --write_json ${OUTPUT_FOLDER}/$out/ \
        --display 0 \
        --render_pose 0 \
        --model_folder $MODEL_FOLDER

    echo "-----------------------------------"
    echo
done

# Example run
# ./run_openpose.sh <input_folder> <output_folder>
