#!/bin/bash

PYPID=`top -n 1 -i -b | grep -i "python" | awk '{print $1}'`
echo -e "$PYPID"

while [[ $(ps -ef | grep -c $PYPID) -ne 1 ]]
do
    top -n 1 -i -b > /tmp/top-lixo

    CPU=`grep -i python /tmp/top-lixo | awk '{print $9}'`
    RUNTIME=`grep -i python /tmp/top-lixo | awk '{print $11}'`
    MEMRES=`grep -i python /tmp/top-lixo | awk '{print $6}'`
    MEMPERC=`grep -i python /tmp/top-lixo | awk '{print $10}'`
    LINE=`cat /proc/$PYPID/fd/1 | wc -l`

    echo Runtime: $RUNTIME CPU: $CPU Memory: $MEMRES  $MEMPERC Breakpoint: $LINE

    sleep 10
done
