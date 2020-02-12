#!/usr/bin/env bash

# wait for Elasticsearch to start, then run the setup script to
# create and configure the index.
exec /utils/wait-for-it.sh localhost:9200 -- /usr/local/bin/elasticsearch/add_index_template.sh &
exec $@