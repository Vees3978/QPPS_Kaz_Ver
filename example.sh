#!/bin/bash

### Displays the job context
echo Running Reconnection Rate QPP analysis
echo Running on host `hostname`
echo Job started at `date +"%T %a %d %b %Y"`
echo Directory is `/veronicaestrada/Downloads/Kazachenko_Lab/Practice_Code/restartable_code_veronica.py`

##############################
### Assigns path variables ###
##############################
i0=0
ie=100
path=`/veronicaestrada/Downloads/Kazachenko_Lab/Practice_Code/restartable_code_veronica.py`
#csv_filename="/Wqpp_Tabel.csv"
#csv_filename2="/No_Wqpp_Tabel.csv"
#filename=${path}${csv_filename}
#filename2=${path}${csv_filename2}

echo Beginning run 'i0 = '${i0} to 'ie = ' ${ie} at `date +"%T %a %d %b %Y"`
echo File will be saved at ${filename} ${filename2}
python restartable_code_veronica.py ${i0} ${ie} {filename} {filename2}
echo Ended run 'i0 = '${i0} to 'ie = ' ${ie} at `date +"%T %a %d %b %Y"`
