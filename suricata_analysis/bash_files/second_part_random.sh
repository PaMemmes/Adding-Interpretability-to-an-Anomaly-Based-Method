#!/bin/bash -x

# Curl files
mkdir ./pcaps

for d in ./splits/split/*; do
        shopt -s nullglob
	numfiles=$(ls "$d" | wc -l)
	filename=$(basename -- "$d")
	echo "$filename"
        waittime=$(echo "scale=4; 0.03*$numfiles" | bc)
	echo "$waittime"
	mkdir ./pcaps/"$filename"
	
        tcpdump -w ./pcaps/"$filename"/"$filename".pcap -i docker0 &
        pid=$!
	sleep 1
	for f in "$d"/*; do
		file=$(basename -- "$f")
                curl http://172.17.0.2:8000/"$filename"/"$file" --output malware

        done
	sleep 1
	kill -2 $pid
done

# Check with Suricata

for d in ./pcaps/*; do
        for f in "$d"/*; do
                echo ./pcaps/"${f##*/}"/"${f##*/}"
                suricata -c /etc/suricata/suricata.yaml -k none -r ./pcaps/"${d##*/}"/"${f##*/}" -l ./pcaps/"${d##*/}" -v
        done
done
