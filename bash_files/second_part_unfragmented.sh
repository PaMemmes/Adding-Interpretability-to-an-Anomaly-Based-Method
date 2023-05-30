#!/bin/bash -x

# Curl files
mkdir /home/pascal/pcaps_unfragmented

for d in /home/pascal/theZoo/split/*; do
        shopt -s nullglob
	
	numfiles=$(ls "$d" | wc -l)
	filename=$(basename -- "$d")
	echo "$filename"
        waittime=$(echo "scale=4; 0.10*$numfiles" | bc)
	mkdir /home/pascal/pcaps_unfragmented/"$filename"
	
        tcpdump -w /home/pascal/pcaps_unfragmented/"$filename"/"$filename".pcap -i docker0 &
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

for d in /home/pascal/pcaps_unfragmented/*; do
        for f in "$d"/*; do
                echo /home/pascal/pcaps_unfragmented/"${f##*/}"/"${f##*/}"
                suricata -c /etc/suricata/suricata.yaml -k none -r /home/pascal/pcaps_unfragmented/"${d##*/}"/"${f##*/}" -l /home/pascal/pcaps_unfragmented/"${d##*/}" -v
        done
done
