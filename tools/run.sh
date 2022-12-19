#!/bin/sh
PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/bin
NUM=`ps -ef | grep "gunicorn_0" | grep -v grep | wc -l`
LOG_NAME=`date +"%m-%d-%Y.%H.%M"`
if [ "$NUM" == "0" ]
  then
     cd /home/robertsone/sail-on-api
     pipenv run gunicorn -c gunicorn_0.config.py 'sail_on.wsgi:create_app()' >> "${LOG_NAME}_unicorn_0.txt" 2>&1
fi
