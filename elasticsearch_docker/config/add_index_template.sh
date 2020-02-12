#!/usr/bin/env bash

# for some reason even when port 9200 is open Elasticsearch is unable to be accessed as authentication fails
# a few seconds later it works
until $(curl -sSf -XGET --insecure --user elastic:changeme 'http://localhost:9200/_cluster/health?wait_for_status=yellow' > /dev/null); do
    printf 'AUTHENTICATION ERROR DUE TO X-PACK, trying again in 5 seconds \n'
    sleep 5
done

templates_file=/usr/local/bin/elasticsearch/templates/*.json

for template in ${templates_file}
do
    full_filename=$(basename -- "$template")
    filename="${full_filename%.*}"
    echo "Add $filename to ES";
    curl -XPUT "localhost:9200/_template/$filename" -H 'Content-Type: application/json' -d @/usr/local/bin/elasticsearch/templates/$full_filename
done