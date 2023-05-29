#!/bin/bash -x

# Curl files
mkdir /home/pascal/pcaps

for d in /home/pascal/splits/split/*; do
        shopt -s nullglob
	numfiles=$(ls "$d" | wc -l)
	filename=$(basename -- "$d")
	echo "$filename"
        waittime=$(echo "scale=4; 0.03*$numfiles" | bc)
	echo "$waittime"
	mkdir /home/pascal/pcaps/"$filename"
	
        tcpdump -w /home/pascal/pcaps/"$filename"/"$filename".pcap -i docker0 &
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
mkdir /home/pascal/suricata_analysis

for d in /home/pascal/pcaps/*; do
        for f in "$d"/*; do
                echo /home/pascal/pcaps/"${f##*/}"/"${f##*/}"
                suricata -c /etc/suricata/suricata.yaml -k none -r /home/pascal/pcaps/"${d##*/}"/"${f##*/}" -l /home/pascal/pcaps/"${d##*/}" -v
        done
done
