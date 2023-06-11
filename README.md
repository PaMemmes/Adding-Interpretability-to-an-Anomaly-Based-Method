# Adding-Interpretability-to-an-Anomaly-Based-Method for Deep Packet Inspection in Intrusion Detection Systems

This github repository includes two different aspects:
1. Suricata analysis and experiment structure in suricata_analysis
2. Machine Learning models in src

First run
``` bash
conda env create -f environment.yml
```
to get all needed dependencies for the python code.

## Suricata Analysis
In suricata_analysis/bash_files bash files can be found with which the fragmented malware was created.
### Prerequisites
* First download github.com/ytisf/thezoo in that directory.
* Install Suricata (https://suricata.io/)
* Configure Suricata (Especially set the HOME_NET variable, etc.)

1. Run bash file with:  
    ```bash
    ./first_part.sh
    ```
2. ```bash 
    cd splits
    ```
    Find IP of docker (if ip addr shows 172.17.0.1 then write 172.17.0.2 in second_part.sh files).
3. ```bash 
    sudo docker build . -t suricata && sudo docker run suricata
    ```
4. After docker is running, go back one dir and run second bash file with:
    ```bash 
    ./second_part.sh
    ```
5. Now in pcaps folder the suricata results and the pcaps have been generated.
   Copy the results, i.e. the pcaps to the suricata_analysis folder.

Repeat these instructions for the unfragmented parts.
Always remove the files that have been created before starting a new experiment since the content may be overridden (i.e. rm -r splits, rm -r theZoo).
Rename the pcaps to theZoo_original or theZoo_fragmented or theZoo_fragmented_random.
Now analysis_comparison.py can be run that creates different plots about the data just created.

## Anomaly-Based methods
To run this part of the project, note that the .csv files need to be created from the .pcap files just created.
They need to be converted using the CICFlowmeterV3.
You can also download these from my [google drive](https://drive.google.com/file/d/1ZEN4pgDDf214EuXuv5biyyLhd9gvwt_D/view?usp=sharing). This needs to be put in the path /mnt/md0/files_memmesheimer/csv_fragmentedV3/.
Please also download the [cicids2018 data set](https://registry.opendata.aws/cse-cic-ids2018/), this needs to be put in path: /mnt/md0/files_memmesheimer/cicids2018/
If they are in a different path, update the filenames in prepocess.py.

Run in src/ :
```bash
    python3 main.py 1 1 1
```

Explanations of the numbers are given in the main.py file.

src/analysis.py creates three different plots of the attack types in csecicids2018, a correlation matrix, provides a ydata ProfileReport for the fragmented data.

