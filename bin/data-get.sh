#!/bin/bash
cd /home/hosokawa/working/php/memo/signal/src/data
PRE=$(git log -1 | grep ^commit | awk '{ print $2 }')
git pull > /dev/null 2>&1
POST=$(git log -1 | grep ^commit | awk '{ print $2 }')
if [ $PRE == $POST ]; then
    echo "Repository was not change."
else
    mkdir /tmp/data-get.lock > /dev/null 2>&1
    if [ $? == 0 ]; then
        ( cd /home/hosokawa/working/troml/; bin/update-ohlc; bin/lstmdata; bin/train10 )
	rmdir /tmp/data-get.lock
    fi
fi
