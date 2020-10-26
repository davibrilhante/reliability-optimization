#! /bin/bash

SPEED=`seq 22 7 64` #30 30 90
BLOCK=`LANG=en_US seq 0.001 0.001 0.010`
AUX=(0.001 0.005 0.010)

for s in ${SPEED[@]}
do
        if [[ $s -eq 22 || $s -eq 43 || $s -eq 64 ]]
        then
                for b in ${BLOCK[@]}
                do
			echo -e "Universe = vanilla\n\
Executable = check-instance.py\n\
Should_transfer_files = YES\n\
\n\
Log = log/$s-$b\n\
Error = err/$s-$b\n\
Out = out/$s-$b\n\
\n\
Transfer_input_files = out/$s/$b\n\
\n\
Arguments = -s $s -b $b\n\
Queue 1" > submit-check.sub

			condor_submit submit-check.sub
		done
	else
		for b in ${AUX[@]}
		do
			echo -e "Universe = vanilla\n\
Executable = check-instance.py\n\
Should_transfer_files = YES\n\
\n\
Log = log/$s-$b\n\
Error = err/$s-$b\n\
Out = out/$s-$b\n\
\n\
Transfer_input_files = out/$s/$b\n\
\n\
Arguments = -s $s -b $b\n\
Queue 1" > submit-check.sub

                        condor_submit submit-check.sub
		done
	fi
done
