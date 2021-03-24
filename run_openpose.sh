INPUT_FOLDER=$1
OUTPUT_FOLDER=$2

OPENPOSE_BIN=/home/gbertamini/openpose/build/examples/openpose/openpose.bin
MODEL_FOLDER=/home/gbertamini/openpose/models/

for i in ${INPUT_FOLDER}/*; do
    echo $i

    # path/to/video_name.mp4
    out=${i%.*}  # path/to/video_name
    out=${out##*/}  # video_name

    $OPENPOSE_BIN \
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
# ./run_openpose.sh /home/galvari/INPUT/Videos/Example_mp4/ /home/galvari/INPUT/JSON/Example_nocopycodec
