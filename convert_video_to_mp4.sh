# convert to mp4 copying codec
for i in Example/*; do
    out=${i%.*}
    out=${out#*/}
    echo ffmpeg -i "$i" -codec copy "Example_mp4_copycodec/${out}_copycodec.mp4"
done

# convert to mp4 without copying codec
for i in Example/*; do
    out=${i%.*}
    out=${out#*/}
    echo ffmpeg -i "$i" "Example_mp4/${out}.mp4"
done
