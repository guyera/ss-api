#!/bin/sh
PATH=/sbin:/bin:/usr/sbin:/usr/bin:/usr/local/bin
NUM=`ps -ef | grep "gunicorn_1.config" | grep -v grep | wc -l`
LOG_NAME=`date +"%m-%d-%Y.%H.%M"`
if [ "$NUM" == "0" ]
  then
     cd /home/robertsone/sail-on-api
     pipenv run gunicorn -c gunicorn_1.config.py 'sail_on.wsgi:create_app()' >> "${LOG_NAME}_unicorn_1.txt" 2>&1
fi
