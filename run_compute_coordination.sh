
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
