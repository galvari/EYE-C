# python src/estimate_coordination.py OUTPUT/JSON_copycodec/1348_high_copycodec_coords.json --input_video OUTPUT/Render_copycodec/1348_high_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/
# python src/estimate_coordination.py OUTPUT/JSON_copycodec/1472_med_copycodec_coords.json --input_video OUTPUT/Render_copycodec/1472_med_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/
# python src/estimate_coordination.py OUTPUT/JSON_copycodec/1507_med_copycodec_coords.json --input_video OUTPUT/Render_copycodec/1507_med_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/
# python src/estimate_coordination.py OUTPUT/JSON_copycodec/1541_high_copycodec_coords.json --input_video OUTPUT/Render_copycodec/1541_high_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/
# python src/estimate_coordination.py OUTPUT/JSON_copycodec/1670_low_copycodec_coords.json --input_video OUTPUT/Render_copycodec/1670_low_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/
# python src/estimate_coordination.py OUTPUT/JSON_copycodec/1730_high_copycodec_coords.json --input_video OUTPUT/Render_copycodec/1730_high_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/
# python src/estimate_coordination.py OUTPUT/JSON_copycodec/1815_med_copycodec_coords.json --input_video OUTPUT/Render_copycodec/1815_med_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/
# python src/estimate_coordination.py OUTPUT/JSON_copycodec/982_low_copycodec_coords.json --input_video OUTPUT/Render_copycodec/982_low_copycodec_gaze.mp4 --output_folder OUTPUT/Render_copycodec/

INPUT_JSON_FOLDER=$1
COORDINATION_FACTOR=$2


for i in ${INPUT_JSON_FOLDER}/*; do
    echo $i


    python src/estimate_coordination.py \
        $i \
        --coordination_factor $COORDINATION_FACTOR
 
    echo "-----------------------------------"
    echo
done

# Example run
# ./run_compute_coordination.sh <input_json_folder> <coordination_factor>
# ./run_compute_coordination.sh ./OUTPUT/JSON_copycodec/ 0.5
